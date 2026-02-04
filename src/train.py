from pathlib import Path

import json
import logging
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from .config import (
    BEST_MODEL_PATH,
    CALIBRATION_METRICS_PATH,
    CONFUSION_MATRIX_PATH,
    FEATURE_IMPORTANCE_CSV,
    FEATURE_IMPORTANCE_PLOT,
    METRICS_PATH,
    METRICS_PLOT_PATH,
    MODELS_DIR,
    ONNX_MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    SHAP_IMPORTANCE_CSV,
    SHAP_SUMMARY_PLOT,
    TRAIN_LOG_PATH,
)
from .data_utils import load_train_test
from .feature_utils import extract_feature_importance, get_feature_names
from .logging_utils import setup_logger
from .model_utils import compute_confusion, compute_metrics, get_models, get_param_grids
from .preprocessing import build_preprocessor, split_features_labels


def train_and_evaluate() -> None:
    logger = setup_logger("train", TRAIN_LOG_PATH)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading NSL-KDD train/test files")
    train_df, test_df = load_train_test()
    X_train, y_train = split_features_labels(train_df)
    X_test, y_test = split_features_labels(test_df)
    logger.info("Train size: %d, Test size: %d", len(X_train), len(X_test))

    models = get_models(RANDOM_STATE)
    param_grids = get_param_grids()

    results = []
    best_model_name = None
    best_model_pipeline = None
    best_f1 = -1.0

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        logger.info("Training %s", name)
        preprocessor = build_preprocessor(X_train)
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        grid = GridSearchCV(
            pipeline,
            param_grids.get(name, {}),
            scoring="f1",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        grid.fit(X_train, y_train)

        best_estimator = grid.best_estimator_
        y_pred = best_estimator.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)
        metrics["model"] = name
        metrics["cv_f1"] = grid.best_score_
        metrics["best_params"] = json.dumps(grid.best_params_)
        results.append(metrics)
        logger.info(
            "%s metrics: acc=%.4f prec=%.4f rec=%.4f f1=%.4f cv_f1=%.4f",
            name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["cv_f1"],
        )

        if metrics["cv_f1"] > best_f1:
            best_f1 = metrics["cv_f1"]
            best_model_name = name
            best_model_pipeline = best_estimator

    if best_model_pipeline is None:
        raise RuntimeError("No model was trained.")

    logger.info("Best model by CV F1: %s (%.4f)", best_model_name, best_f1)

    calibrated_model, threshold, calibration_metrics = _calibrate_model(
        best_model_pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    calibrated_confusion = _confusion_with_threshold(calibrated_model, X_test, y_test, threshold)

    joblib.dump(
        {
            "model": calibrated_model,
            "best_model": best_model_name,
            "threshold": threshold,
        },
        BEST_MODEL_PATH,
    )
    logger.info("Saved calibrated model to %s", BEST_MODEL_PATH)

    metrics_df = pd.DataFrame(results).set_index("model").sort_values(by="f1", ascending=False)
    metrics_df.to_csv(METRICS_PATH)
    logger.info("Saved metrics to %s", METRICS_PATH)

    _plot_metrics(metrics_df, METRICS_PLOT_PATH)
    _plot_confusion(calibrated_confusion, CONFUSION_MATRIX_PATH, best_model_name)
    logger.info("Saved plots to %s and %s", METRICS_PLOT_PATH, CONFUSION_MATRIX_PATH)

    metrics_json_path = REPORTS_DIR / "metrics.json"
    _save_metrics_json(metrics_df, best_model_name, metrics_json_path)
    logger.info("Saved metrics JSON to %s", metrics_json_path)

    _save_calibration_metrics(
        calibration_metrics,
        threshold,
        best_model_name,
        CALIBRATION_METRICS_PATH,
    )
    logger.info("Saved calibration metrics to %s", CALIBRATION_METRICS_PATH)
    _save_feature_importance(
        best_model_pipeline,
        X_test,
        y_test,
        FEATURE_IMPORTANCE_CSV,
        FEATURE_IMPORTANCE_PLOT,
    )
    logger.info("Saved feature importance to %s", FEATURE_IMPORTANCE_CSV)
    _save_shap_summary(
        best_model_pipeline,
        X_train,
        SHAP_IMPORTANCE_CSV,
        SHAP_SUMMARY_PLOT,
    )
    logger.info("Saved SHAP reports to %s", SHAP_IMPORTANCE_CSV)
    _export_onnx(best_model_pipeline, X_train, ONNX_MODEL_PATH)
    logger.info("ONNX export attempted to %s", ONNX_MODEL_PATH)



def _plot_metrics(metrics_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = metrics_df[["accuracy", "precision", "recall", "f1"]]
    ax = plot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def _plot_confusion(confusion, out_path: Path, model_name: str) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=["Normal", "Attack"])
    disp.plot(values_format="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_metrics_json(metrics_df: pd.DataFrame, best_model_name: str, out_path: Path) -> None:
    payload = {
        "best_model": best_model_name,
        "models": metrics_df.reset_index().to_dict(orient="records"),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _confusion_with_threshold(model, X, y, threshold: float):
    try:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1]
            preds = (scores >= threshold).astype(int)
        else:
            preds = model.predict(X)
    except Exception:
        preds = model.predict(X)
    return compute_confusion(y, preds)


def _calibrate_model(best_model_pipeline, X_train, y_train, X_test, y_test):
    logger = logging.getLogger("train")
    try:
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )
        base_model = clone(best_model_pipeline)
        base_model.fit(X_fit, y_fit)

        try:
            calibrator = CalibratedClassifierCV(estimator=base_model, cv="prefit", method="sigmoid")
        except TypeError:
            calibrator = CalibratedClassifierCV(base_estimator=base_model, cv="prefit", method="sigmoid")
        calibrator.fit(X_cal, y_cal)

        cal_probs = calibrator.predict_proba(X_cal)[:, 1]
        threshold = _find_best_threshold(y_cal, cal_probs)

        test_probs = calibrator.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= threshold).astype(int)
        metrics = compute_metrics(y_test, test_preds)
        metrics["threshold"] = threshold
        logger.info(
            "Calibration complete. Threshold=%.3f, test_f1=%.4f",
            threshold,
            metrics.get("f1", -1.0),
        )
        return calibrator, threshold, metrics
    except Exception as exc:
        warnings.warn(f"Calibration failed, using uncalibrated model. Reason: {exc}")
        logger.warning("Calibration failed: %s", exc)
        fallback_preds = best_model_pipeline.predict(X_test)
        metrics = compute_metrics(y_test, fallback_preds)
        metrics["threshold"] = 0.5
        return best_model_pipeline, 0.5, metrics


def _find_best_threshold(y_true, scores):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return float(best_threshold)


def _save_calibration_metrics(metrics: dict, threshold: float, model_name: str, out_path: Path) -> None:
    logger = logging.getLogger("train")
    payload = {
        "best_model": model_name,
        "threshold": threshold,
        "metrics": metrics,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Calibration metrics saved")


def _save_feature_importance(pipeline, X_test, y_test, csv_path: Path, plot_path: Path) -> None:
    logger = logging.getLogger("train")
    feature_names, importances = extract_feature_importance(pipeline)

    if feature_names is None:
        try:
            perm = permutation_importance(
                pipeline,
                X_test,
                y_test,
                n_repeats=5,
                random_state=RANDOM_STATE,
                scoring="f1",
                n_jobs=1,
            )
            feature_names = list(X_test.columns)
            importances = perm.importances_mean
        except Exception as exc:
            warnings.warn(f"Feature importance unavailable: {exc}")
            logger.warning("Feature importance unavailable: %s", exc)
            csv_path.write_text("Feature importance unavailable.", encoding="utf-8")
            return

    if len(feature_names) != len(importances):
        warnings.warn("Feature importance length mismatch; skipping.")
        return

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values(by="importance", ascending=False)
    df.to_csv(csv_path, index=False)
    logger.info("Feature importance CSV saved")

    plot_df = df.head(20)
    ax = plot_df.plot(kind="barh", x="feature", y="importance", figsize=(10, 7), legend=False)
    ax.invert_yaxis()
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    logger.info("Feature importance plot saved")


def _save_shap_summary(pipeline, X_train, csv_path: Path, plot_path: Path) -> None:
    logger = logging.getLogger("train")
    try:
        import shap
    except Exception as exc:
        _write_error(REPORTS_DIR / "shap_error.txt", f"SHAP not installed: {exc}")
        logger.warning("SHAP not installed")
        return

    if not hasattr(pipeline, "named_steps"):
        _write_error(REPORTS_DIR / "shap_error.txt", "Pipeline missing named_steps.")
        return

    preprocessor = pipeline.named_steps.get("preprocess")
    model = pipeline.named_steps.get("model")
    if preprocessor is None or model is None:
        _write_error(REPORTS_DIR / "shap_error.txt", "Pipeline missing preprocess/model steps.")
        return

    sample_size = min(200, len(X_train))
    if sample_size < 2:
        _write_error(REPORTS_DIR / "shap_error.txt", "Not enough samples for SHAP.")
        return

    X_sample = X_train.sample(n=sample_size, random_state=RANDOM_STATE)
    X_trans = preprocessor.transform(X_sample)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    feature_names = get_feature_names(preprocessor)
    if len(feature_names) != X_trans.shape[1]:
        feature_names = [f"f{i}" for i in range(X_trans.shape[1])]

    model_name = model.__class__.__name__.lower()
    try:
        if "forest" in model_name or "tree" in model_name:
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, X_trans, feature_names=feature_names)
        else:
            background = shap.sample(X_trans, min(50, X_trans.shape[0]))
            if hasattr(model, "predict_proba"):
                predict_fn = model.predict_proba
            else:
                predict_fn = model.predict
            explainer = shap.KernelExplainer(predict_fn, background)

        shap_values = explainer.shap_values(X_trans)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        shap_df = shap_df.sort_values(by="mean_abs_shap", ascending=False)
        shap_df.to_csv(csv_path, index=False)

        shap.summary_plot(
            shap_values,
            X_trans,
            feature_names=feature_names,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
    except Exception as exc:
        _write_error(REPORTS_DIR / "shap_error.txt", f"SHAP failed: {exc}")
        logger.warning("SHAP failed: %s", exc)


def _export_onnx(pipeline, X_train, out_path: Path) -> None:
    logger = logging.getLogger("train")
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType
        from .config import CATEGORICAL_COLS
    except Exception as exc:
        _write_error(REPORTS_DIR / "onnx_export_error.txt", f"ONNX export unavailable: {exc}")
        logger.warning("ONNX export unavailable: %s", exc)
        return

    initial_types = []
    for col in X_train.columns:
        if col in CATEGORICAL_COLS:
            initial_types.append((col, StringTensorType([None, 1])))
        else:
            initial_types.append((col, FloatTensorType([None, 1])))

    try:
        onnx_model = convert_sklearn(pipeline, initial_types=initial_types)
        out_path.write_bytes(onnx_model.SerializeToString())
    except Exception as exc:
        _write_error(REPORTS_DIR / "onnx_export_error.txt", f"ONNX export failed: {exc}")
        logger.warning("ONNX export failed: %s", exc)


def _write_error(path: Path, message: str) -> None:
    path.write_text(message, encoding="utf-8")


if __name__ == "__main__":
    train_and_evaluate()

