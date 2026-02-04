import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

from .config import (
    BEST_MODEL_PATH,
    COLUMNS,
    DROP_COLS,
    LIVE_ALERTS_PATH,
    LIVE_LOG_PATH,
    TARGET_COL,
)
from .logging_utils import setup_logger
from .live_utils import compute_flow_features, expected_feature_columns, safe_float, safe_int

WINDOW_SIZE = 100


def _load_model():
    bundle = joblib.load(BEST_MODEL_PATH)
    return bundle["model"], bundle.get("best_model"), bundle.get("threshold", 0.5)


def _get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X)
        return 1 / (1 + np.exp(-raw_scores))
    return None


def _parse_zeek_tsv(line: str, fields: List[str]) -> Optional[Dict[str, str]]:
    if not fields:
        return None
    parts = line.split("	")
    if len(parts) < len(fields):
        return None
    return dict(zip(fields, parts))


def _parse_zeek_json(line: str) -> Optional[Dict[str, str]]:
    try:
        data = json.loads(line)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _normalize_row(row: Dict[str, str]) -> Dict[str, str]:
    return row


def _tail_lines(path: Path, start_at_end: bool) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        if start_at_end:
            handle.seek(0, 2)
        while True:
            line = handle.readline()
            if not line:
                time.sleep(0.5)
                continue
            yield line.rstrip("
")


def run_live(log_path: Path, output_path: Path, threshold: Optional[float], start_at_end: bool) -> None:
    logger = setup_logger("live_ids", LIVE_LOG_PATH)
    model, model_name, model_threshold = _load_model()
    threshold = model_threshold if threshold is None else threshold

    logger.info("Starting live IDS on %s", log_path)
    logger.info("Model: %s | Threshold: %.3f", model_name, threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()

    expected_cols = expected_feature_columns(COLUMNS, DROP_COLS, TARGET_COL)
    recent = deque(maxlen=WINDOW_SIZE)
    fields: List[str] = []

    with output_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["timestamp", "src", "dst", "service", "score", "prediction"])
        if not file_exists:
            writer.writeheader()

        for line in _tail_lines(log_path, start_at_end):
            if not line:
                continue
            if line.startswith("#fields"):
                fields = line.split()[1:]
                logger.info("Detected Zeek fields: %s", ", ".join(fields))
                continue
            if line.startswith("#"):
                continue

            if line.lstrip().startswith("{"):
                row = _parse_zeek_json(line)
                if row is None:
                    continue
                row = _normalize_row(row)
            else:
                row = _parse_zeek_tsv(line, fields)
                if row is None:
                    continue

            src = row.get("id.orig_h", "")
            dst = row.get("id.resp_h", "")
            service = row.get("service", "unknown")
            proto = row.get("proto", "tcp")
            flag = row.get("conn_state", "OTH")
            src_port = safe_int(row.get("id.orig_p", 0))

            duration = safe_float(row.get("duration", 0.0))
            src_bytes = safe_int(row.get("orig_bytes", 0))
            dst_bytes = safe_int(row.get("resp_bytes", 0))

            feature_row = compute_flow_features(
                src=src,
                dst=dst,
                service=service,
                proto=proto,
                flag=flag,
                src_port=src_port,
                duration=duration,
                src_bytes=src_bytes,
                dst_bytes=dst_bytes,
                recent=recent,
            )

            feature_df = pd.DataFrame([feature_row], columns=expected_cols)
            scores = _get_scores(model, feature_df)
            if scores is not None:
                score = float(scores[0])
                pred = 1 if score >= threshold else 0
            else:
                pred = int(model.predict(feature_df)[0])
                score = float(pred)

            timestamp = row.get("ts", "")

            writer.writerow(
                {
                    "timestamp": timestamp,
                    "src": src,
                    "dst": dst,
                    "service": service,
                    "score": round(score, 4),
                    "prediction": "attack" if pred == 1 else "normal",
                }
            )
            csvfile.flush()

            if pred == 1:
                logger.warning("ALERT src=%s dst=%s service=%s score=%.3f", src, dst, service, score)
            else:
                logger.info("OK src=%s dst=%s service=%s score=%.3f", src, dst, service, score)


def main():
    parser = argparse.ArgumentParser(description="Live IDS from Zeek conn.log")
    parser.add_argument("--log", required=True, help="Path to Zeek conn.log")
    parser.add_argument(
        "--output",
        default=str(LIVE_ALERTS_PATH),
        help="Path to save live alerts CSV",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold (defaults to trained threshold)",
    )
    parser.add_argument(
        "--start-at-beginning",
        action="store_true",
        help="Process existing log from start (default: tail new lines)",
    )

    args = parser.parse_args()
    run_live(Path(args.log), Path(args.output), args.threshold, args.start_at_beginning)


if __name__ == "__main__":
    main()
