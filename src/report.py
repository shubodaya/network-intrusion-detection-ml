import argparse
import json
from pathlib import Path

import pandas as pd

from .config import REPORTS_DIR


def generate_report(metrics_csv: Path, output_json: Path) -> None:
    df = pd.read_csv(metrics_csv)
    if "model" not in df.columns:
        raise ValueError("metrics.csv must contain a 'model' column")

    df = df.sort_values(by="f1", ascending=False)
    best_model = df.iloc[0]["model"]

    print("Model Performance (sorted by F1):")
    print(df[["model", "accuracy", "precision", "recall", "f1", "cv_f1"]].to_string(index=False))
    print(f"\nBest model: {best_model}")

    payload = {
        "best_model": best_model,
        "models": df.to_dict(orient="records"),
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved JSON report to: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print metrics and save JSON summary")
    parser.add_argument(
        "--metrics",
        default=str(REPORTS_DIR / "metrics.csv"),
        help="Path to metrics CSV",
    )
    parser.add_argument(
        "--output",
        default=str(REPORTS_DIR / "metrics.json"),
        help="Path to output JSON",
    )
    args = parser.parse_args()

    generate_report(Path(args.metrics), Path(args.output))

