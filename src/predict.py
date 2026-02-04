import argparse
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from .config import BEST_MODEL_PATH, COLUMNS, DROP_COLS, TARGET_COL, PREDICT_LOG_PATH, REPORTS_DIR
from .logging_utils import setup_logger


def _expected_feature_columns():
    return [c for c in COLUMNS if c not in DROP_COLS + [TARGET_COL]]


def load_model(model_path: Path):
    bundle = joblib.load(model_path)
    return bundle["model"], bundle.get("best_model"), bundle.get("threshold", 0.5)


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    expected_cols = _expected_feature_columns()
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(missing)
        )

    df = df[expected_cols]
    return df


def _get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X)
        return 1 / (1 + np.exp(-raw_scores))
    return None


def _summary_stats(df: pd.DataFrame, preds: np.ndarray) -> dict:
    total = int(len(preds))
    attack_count = int(preds.sum())
    normal_count = int(total - attack_count)
    summary = {
        "total_rows": total,
        "attack_count": attack_count,
        "normal_count": normal_count,
        "attack_rate": attack_count / total if total else 0.0,
    }

    for col in ["protocol_type", "service", "flag"]:
        if col in df.columns:
            summary[f"top_{col}"] = df[col].value_counts().head(5).to_dict()
            summary[f"{col}_attack_rate"] = (
                df.assign(pred=preds)
                .groupby(col)["pred"]
                .mean()
                .sort_values(ascending=False)
                .head(5)
                .to_dict()
            )

    return summary


def predict(input_csv: Path, output_csv: Path, threshold: Optional[float], summary_path: Optional[Path]) -> None:
    logger = setup_logger("predict", PREDICT_LOG_PATH)
    model, best_model_name, model_threshold = load_model(BEST_MODEL_PATH)

    df = pd.read_csv(input_csv)
    X = prepare_input(df)
    logger.info("Loaded input: %s (%d rows)", input_csv, len(df))

    scores = _get_scores(model, X)
    threshold = model_threshold if threshold is None else threshold

    if scores is not None:
        preds = (scores >= threshold).astype(int)
    else:
        preds = model.predict(X)

    pred_labels = ["normal" if p == 0 else "attack" for p in preds]

    out_df = df.copy()
    out_df["prediction"] = preds
    out_df["prediction_label"] = pred_labels
    if scores is not None:
        out_df["prediction_score"] = scores

    out_df.to_csv(output_csv, index=False)

    if best_model_name:
        print(f"Model used: {best_model_name}")
        logger.info("Model used: %s", best_model_name)
    if scores is not None:
        print(f"Decision threshold: {threshold}")
        logger.info("Decision threshold: %.4f", threshold)
    print(f"Predictions saved to: {output_csv}")
    logger.info("Predictions saved to: %s", output_csv)

    if summary_path is not None:
        summary = _summary_stats(df, preds)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary saved to: {summary_path}")
        logger.info("Summary saved to: %s", summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict intrusions from network traffic CSV")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for attack (defaults to trained threshold)",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional path to save a JSON summary report",
    )

    args = parser.parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary) if args.summary else None
    predict(Path(args.input), Path(args.output), args.threshold, summary_path)
