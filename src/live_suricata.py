import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd

from .config import BEST_MODEL_PATH, LIVE_ALERTS_PATH, LIVE_LOG_PATH, COLUMNS, DROP_COLS, TARGET_COL
from .logging_utils import setup_logger
from .live_utils import compute_flow_features, expected_feature_columns, safe_int, safe_float

WINDOW_SIZE = 200


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


def _parse_flow_event(evt: dict) -> Optional[dict]:
    if not isinstance(evt, dict):
        return None
    if evt.get("event_type") not in {"flow", "netflow"}:
        return None

    src = evt.get("src_ip") or evt.get("src") or ""
    dst = evt.get("dest_ip") or evt.get("dest") or ""
    src_port = safe_int(evt.get("src_port") or evt.get("sport") or 0)
    dst_port = safe_int(evt.get("dest_port") or evt.get("dport") or 0)
    proto = str(evt.get("proto") or evt.get("protocol") or "tcp").lower()

    flow = evt.get("flow", {}) if isinstance(evt.get("flow"), dict) else {}
    duration = safe_float(flow.get("duration", 0.0))
    src_bytes = safe_int(flow.get("bytes_toserver", 0))
    dst_bytes = safe_int(flow.get("bytes_toclient", 0))

    service = evt.get("app_proto") or evt.get("app_proto_ts") or "unknown"
    conn_state = flow.get("state", "SF")

    return {
        "timestamp": evt.get("timestamp", ""),
        "src": src,
        "dst": dst,
        "src_port": src_port,
        "dst_port": dst_port,
        "proto": proto,
        "service": service,
        "flag": conn_state,
        "duration": duration,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
    }


def run_live(log_path: Path, output_path: Path, threshold: Optional[float], start_at_end: bool) -> None:
    logger = setup_logger("live_suricata", LIVE_LOG_PATH)
    model, model_name, model_threshold = _load_model()
    threshold = model_threshold if threshold is None else threshold

    logger.info("Starting Suricata live IDS on %s", log_path)
    logger.info("Model: %s | Threshold: %.3f", model_name, threshold)

    expected_cols = expected_feature_columns(COLUMNS, DROP_COLS, TARGET_COL)
    recent = deque(maxlen=WINDOW_SIZE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()

    with output_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["timestamp", "src", "dst", "service", "score", "prediction"],
        )
        if not file_exists:
            writer.writeheader()

        for line in _tail_lines(log_path, start_at_end):
            if not line:
                continue
            try:
                evt = json.loads(line)
            except Exception:
                continue

            parsed = _parse_flow_event(evt)
            if parsed is None:
                continue

            feature_row = compute_flow_features(
                src=parsed["src"],
                dst=parsed["dst"],
                service=parsed["service"],
                proto=parsed["proto"],
                flag=parsed["flag"],
                src_port=parsed["src_port"],
                duration=parsed["duration"],
                src_bytes=parsed["src_bytes"],
                dst_bytes=parsed["dst_bytes"],
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

            writer.writerow(
                {
                    "timestamp": parsed["timestamp"],
                    "src": parsed["src"],
                    "dst": parsed["dst"],
                    "service": parsed["service"],
                    "score": round(score, 4),
                    "prediction": "attack" if pred == 1 else "normal",
                }
            )
            csvfile.flush()

            if pred == 1:
                logger.warning("ALERT src=%s dst=%s service=%s score=%.3f", parsed["src"], parsed["dst"], parsed["service"], score)
            else:
                logger.info("OK src=%s dst=%s service=%s score=%.3f", parsed["src"], parsed["dst"], parsed["service"], score)


def main():
    parser = argparse.ArgumentParser(description="Live IDS from Suricata eve.json (flow events)")
    parser.add_argument("--log", required=True, help="Path to Suricata eve.json")
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
