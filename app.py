from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.config import BEST_MODEL_PATH, COLUMNS, DROP_COLS, TARGET_COL, FEATURE_IMPORTANCE_CSV
from src.predict import prepare_input


st.set_page_config(page_title="NIDS Predictor", layout="wide")

st.title("Network Intrusion Detection System")
st.write("Upload a CSV of network traffic and get intrusion predictions.")


@st.cache_resource
def load_model():
    import joblib

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {BEST_MODEL_PATH}. Run: python -m src.train"
        )
    bundle = joblib.load(BEST_MODEL_PATH)
    return bundle["model"], bundle.get("best_model"), bundle.get("threshold", 0.5)


def binarize_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    return series.apply(lambda x: 0 if str(x).strip() == "normal" else 1).astype(int)


def get_feature_names(preprocessor):
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        pass

    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "steps"):
            transformer = transformer.steps[-1][1]

        if hasattr(transformer, "get_feature_names_out"):
            try:
                names = transformer.get_feature_names_out(cols)
            except TypeError:
                names = transformer.get_feature_names_out()
            feature_names.extend(list(names))
        elif hasattr(transformer, "get_feature_names"):
            try:
                names = transformer.get_feature_names(cols)
            except TypeError:
                names = transformer.get_feature_names()
            feature_names.extend(list(names))
        else:
            if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                feature_names.extend(list(cols))
            else:
                feature_names.append(cols)
    return feature_names


def extract_feature_importance(pipeline):
    if not hasattr(pipeline, "named_steps"):
        return None, None
    preprocessor = pipeline.named_steps.get("preprocess")
    model = pipeline.named_steps.get("model")
    if preprocessor is None or model is None:
        return None, None

    feature_names = get_feature_names(preprocessor)
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()

    if importances is None:
        return None, None

    if len(feature_names) != len(importances):
        return None, None
    return feature_names, importances


def load_saved_feature_importance():
    if not FEATURE_IMPORTANCE_CSV.exists():
        return None
    try:
        df = pd.read_csv(FEATURE_IMPORTANCE_CSV)
        if "feature" in df.columns and "importance" in df.columns:
            return df
    except Exception:
        return None
    return None


model, model_name, default_threshold = None, None, 0.5
try:
    model, model_name, default_threshold = load_model()
    if model_name:
        st.success(f"Loaded model: {model_name}")
    else:
        st.success("Loaded model")
except Exception as exc:
    st.error(str(exc))
tab_upload, tab_live = st.tabs(["Upload CSV", "Live IDS"])

with tab_upload:
    st.subheader("Upload CSV")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    
    threshold = st.slider(
        "Decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(default_threshold),
        step=0.01,
    )
    
    if uploaded and model is not None:
        df = pd.read_csv(uploaded)
        try:
            X = prepare_input(df)
        except Exception as exc:
            st.error(str(exc))
        else:
            scores = None
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                raw_scores = model.decision_function(X)
                scores = 1 / (1 + np.exp(-raw_scores))
    
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
    
            st.subheader("Predictions")
            st.write(out_df.head(20))
    
            counts = pd.Series(pred_labels).value_counts()
            if not counts.empty:
                st.bar_chart(counts)
            else:
                st.info("No predictions to chart yet.")
    
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download predictions CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
            )
    
            st.subheader("Per-Class Precision/Recall")
            if TARGET_COL in df.columns:
                y_true = binarize_labels(df[TARGET_COL])
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, preds, labels=[0, 1], zero_division=0
                )
                metrics_df = pd.DataFrame(
                    {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    },
                    index=["normal", "attack"],
                )
                st.dataframe(metrics_df)
                chart_df = metrics_df[["precision", "recall"]].dropna()
                if not chart_df.empty:
                    st.bar_chart(chart_df)
                else:
                    st.info("Precision/recall chart unavailable.")
    
                cm = confusion_matrix(y_true, preds, labels=[0, 1])
                st.write("Confusion Matrix")
                st.dataframe(
                    pd.DataFrame(
                        cm,
                        index=["true_normal", "true_attack"],
                        columns=["pred_normal", "pred_attack"],
                    )
                )
            else:
                st.info("Upload a CSV with a 'label' column to compute precision/recall.")
    
            st.subheader("Feature Importance")
            use_uploaded = st.checkbox("Compute importance from uploaded data (slower)", value=False)
    
            if use_uploaded:
                if TARGET_COL not in df.columns:
                    st.info("Upload a CSV with a 'label' column to compute data-specific importance.")
                else:
                    y_true = binarize_labels(df[TARGET_COL])
                    X_imp = prepare_input(df)
                    sample_n = min(len(X_imp), 1000)
                    if sample_n < len(X_imp):
                        X_imp = X_imp.sample(sample_n, random_state=42)
                        y_true = y_true.loc[X_imp.index]
    
                    with st.spinner("Computing permutation importance..."):
                        try:
                            perm = permutation_importance(
                                model,
                                X_imp,
                                y_true,
                                n_repeats=3,
                                random_state=42,
                                scoring="f1",
                                n_jobs=1,
                            )
                            feature_names, _ = extract_feature_importance(model)
                            if feature_names is None:
                                feature_names = list(X_imp.columns)
    
                            imp_df = (
                                pd.DataFrame(
                                    {"feature": feature_names, "importance": perm.importances_mean}
                                )
                                .sort_values(by="importance", ascending=False)
                                .head(20)
                            )
                            st.dataframe(imp_df)
                            chart_df = imp_df.set_index("feature").dropna()
                            if not chart_df.empty:
                                st.bar_chart(chart_df)
                            else:
                                st.info("Feature importance chart unavailable.")
                        except Exception as exc:
                            st.info(f"Unable to compute permutation importance: {exc}")
            else:
                feature_names, importances = extract_feature_importance(model)
                if feature_names is None:
                    saved = load_saved_feature_importance()
                    if saved is None:
                        st.info("Feature importance is not available for this model.")
                    else:
                        imp_df = saved.head(20)
                        st.dataframe(imp_df)
                        chart_df = imp_df.set_index("feature").dropna()
                        if not chart_df.empty:
                            st.bar_chart(chart_df)
                        else:
                            st.info("Feature importance chart unavailable.")
                else:
                    imp_df = (
                        pd.DataFrame({"feature": feature_names, "importance": importances})
                        .sort_values(by="importance", ascending=False)
                        .head(20)
                    )
                    st.dataframe(imp_df)
                    chart_df = imp_df.set_index("feature").dropna()
                    if not chart_df.empty:
                        st.bar_chart(chart_df)
                    else:
                        st.info("Feature importance chart unavailable.")
    
    st.subheader("Expected Columns")
    expected_cols = [c for c in COLUMNS if c not in DROP_COLS + [TARGET_COL]]
    st.code(", ".join(expected_cols))
    
with tab_live:
    st.subheader("Live IDS Controls")
    st.write("Start one live mode at a time from the app. Use separate terminals if you want multiple modes.")
    
    if "live_proc" not in st.session_state:
        st.session_state.live_proc = None
    if "live_cmd" not in st.session_state:
        st.session_state.live_cmd = None
    
    mode = st.selectbox(
        "Live Mode",
        ["Live CSV (NSL-KDD columns)", "Windows Sysmon", "Zeek conn.log", "Suricata eve.json"],
    )
    
    use_model_threshold = st.checkbox("Use model threshold", value=True)
    custom_threshold = st.number_input("Custom threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    threshold_arg = None if use_model_threshold else float(custom_threshold)
    
    start_at_beginning = st.checkbox("Process from beginning", value=False)
    
    cmd = None
    output_hint = None
    
    if mode == "Live CSV (NSL-KDD columns)":
        csv_path = st.text_input("Input CSV path", value="data\\sample_input.csv")
        output_path = st.text_input("Output CSV path", value="reports\\live_predictions.csv")
        cmd = [sys.executable, "-m", "src.live_csv", "--input", csv_path, "--output", output_path]
        output_hint = output_path
        if threshold_arg is not None:
            cmd += ["--threshold", str(threshold_arg)]
        if start_at_beginning:
            cmd += ["--start-at-beginning"]
    elif mode == "Windows Sysmon":
        channel = st.text_input("Event log channel", value="Microsoft-Windows-Sysmon/Operational")
        event_id = st.number_input("Event ID", min_value=1, max_value=65535, value=3, step=1)
        output_path = st.text_input("Output CSV path", value="reports\\live_alerts_sysmon.csv")
        st.info("Requires Sysmon installed and the pywin32 package.")
        cmd = [
            sys.executable,
            "-m",
            "src.live_windows",
            "--channel",
            channel,
            "--event-id",
            str(int(event_id)),
            "--output",
            output_path,
        ]
        output_hint = output_path
        if threshold_arg is not None:
            cmd += ["--threshold", str(threshold_arg)]
        if not start_at_beginning:
            cmd += ["--start-at-end"]
    elif mode == "Zeek conn.log":
        log_path = st.text_input("conn.log path", value="C:\\path\\to\\conn.log")
        output_path = st.text_input("Output CSV path", value="reports\\live_alerts.csv")
        cmd = [sys.executable, "-m", "src.live_ids", "--log", log_path, "--output", output_path]
        output_hint = output_path
        if threshold_arg is not None:
            cmd += ["--threshold", str(threshold_arg)]
        if start_at_beginning:
            cmd += ["--start-at-beginning"]
    else:
        log_path = st.text_input("eve.json path", value="C:\\path\\to\\eve.json")
        output_path = st.text_input("Output CSV path", value="reports\\live_alerts_suricata.csv")
        cmd = [sys.executable, "-m", "src.live_suricata", "--log", log_path, "--output", output_path]
        output_hint = output_path
        if threshold_arg is not None:
            cmd += ["--threshold", str(threshold_arg)]
        if start_at_beginning:
            cmd += ["--start-at-beginning"]
    
    col_start, col_stop = st.columns(2)
    
    with col_start:
        if st.button("Start Live IDS"):
            if st.session_state.live_proc is not None and st.session_state.live_proc.poll() is None:
                st.warning("A live process is already running. Stop it first.")
            else:
                if cmd is None:
                    st.error("Invalid configuration.")
                else:
                    st.session_state.live_proc = subprocess.Popen(
                        cmd,
                        cwd=Path(__file__).resolve().parent,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    st.session_state.live_cmd = " ".join(cmd)
                    st.success("Live IDS started.")
    
    with col_stop:
        if st.button("Stop Live IDS"):
            proc = st.session_state.live_proc
            if proc is None or proc.poll() is not None:
                st.info("No live process is running.")
            else:
                proc.terminate()
                st.success("Live IDS stopped.")
    
    proc = st.session_state.live_proc
    if proc is not None and proc.poll() is None:
        st.info(f"Running: {st.session_state.live_cmd}")
    else:
        st.info("No live process running.")
    
    if output_hint:
        output_file = Path(output_hint)
        if output_file.exists():
            st.write("Latest alerts (last 20 rows)")
            try:
                alert_df = pd.read_csv(output_file).tail(20)
                st.dataframe(alert_df)
            except Exception:
                st.info("Unable to read alerts file yet.")
