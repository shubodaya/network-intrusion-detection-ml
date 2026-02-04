import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

from .config import BEST_MODEL_PATH, COLUMNS, DROP_COLS, TARGET_COL, PREDICT_LOG_PATH
from .logging_utils import setup_logger
from .live_utils import expected_feature_columns


SLEEP_SECONDS = 0.5


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


def _wait_for_file(path: Path):
    while not path.exists():
        time.sleep(SLEEP_SECONDS)


def _tail_csv_rows(path: Path, start_at_end: bool) -> Iterable[Dict[str, str]]:
    _wait_for_file(path)
    with path.open("r", newline="", encoding="utf-8", errors="ignore") as handle:
        header_line = handle.readline()
        while not header_line:
            time.sleep(SLEEP_SECONDS)
            header_line = handle.readline()
        fieldnames = next(csv.reader([header_line]))

        if start_at_end:
            handle.seek(0, 2)

        while True:
            position = handle.tell()
            line = handle.readline()
            if not line:
                time.sleep(SLEEP_SECONDS)
                handle.seek(position)
                continue
            row = next(csv.reader([line]))
            if len(row) != len(fieldnames):
                continue
            yield dict(zip(fieldnames, row))


def run_live(input_csv: Path, output_csv: Path, threshold: Optional[float], start_at_end: bool) -> None:
    logger = setup_logger("live_csv", PREDICT_LOG_PATH)
    model, model_name, model_threshold = _load_model()
    threshold = model_threshold if threshold is None else threshold

    logger.info("Starting live CSV IDS on %s", input_csv)
    logger.info("Model: %s | Threshold: %.3f", model_name, threshold)

    expected_cols = expected_feature_columns(COLUMNS, DROP_COLS, TARGET_COL)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()

    with output_csv.open("a", newline="", encoding="utf-8") as outfile:
        writer = None
        for row in _tail_csv_rows(input_csv, start_at_end):
            missing = [c for c in expected_cols if c not in row]
            if missing:
                logger.warning("Skipping row; missing columns: %s", ", ".join(missing))
                continue

            feature_row = {c: row.get(c, "") for c in expected_cols}
            feature_df = pd.DataFrame([feature_row], columns=expected_cols)

            scores = _get_scores(model, feature_df)
            if scores is not None:
                score = float(scores[0])
                pred = 1 if score >= threshold else 0
            else:
                pred = int(model.predict(feature_df)[0])
                score = float(pred)

            row_out = dict(row)
            row_out["prediction"] = pred
            row_out["prediction_label"] = "attack" if pred == 1 else "normal"
            row_out["prediction_score"] = round(score, 4)

            if writer is None:
                fieldnames = list(row_out.keys())
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()

            writer.writerow(row_out)
            outfile.flush()
            if pred == 1:
                logger.warning("ALERT score=%.3f", score)
            else:
                logger.info("OK score=%.3f", score)


def main():
    parser = argparse.ArgumentParser(description="Live IDS by tailing a CSV with NSL-KDD columns")
    parser.add_argument("--input", required=True, help="Path to CSV to tail")
    parser.add_argument(
        "--output",
        default="reports/live_predictions.csv",
        help="Path to save predictions CSV",
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
        help="Process existing rows from start (default: tail new rows)",
    )

    args = parser.parse_args()
    run_live(Path(args.input), Path(args.output), args.threshold, args.start_at_beginning)


if __name__ == "__main__":
    main()
