import argparse
import csv
import time
from collections import deque
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from .config import BEST_MODEL_PATH, LIVE_LOG_PATH, LIVE_ALERTS_PATH, COLUMNS, DROP_COLS, TARGET_COL
from .logging_utils import setup_logger
from .live_utils import compute_flow_features, expected_feature_columns, safe_int

WINDOW_SIZE = 200
POLL_SECONDS = 1.0

PORT_SERVICE = {
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    53: "dns",
    80: "http",
    110: "pop3",
    143: "imap",
    443: "https",
    445: "microsoft-ds",
    3389: "rdp",
}


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


def _map_service(port: int) -> str:
    return PORT_SERVICE.get(port, "other")


def _parse_sysmon_xml(xml: str) -> Optional[dict]:
    try:
        import xml.etree.ElementTree as ET
    except Exception:
        return None

    try:
        root = ET.fromstring(xml)
    except Exception:
        return None

    ns = {"e": "http://schemas.microsoft.com/win/2004/08/events/event"}
    data_nodes = root.findall(".//e:EventData/e:Data", ns)
    data = {}
    for node in data_nodes:
        key = (node.attrib.get("Name") or "").lower()
        val = node.text or ""
        if key:
            data[key] = val

    system = root.find("e:System", ns)
    timestamp = ""
    if system is not None:
        time_node = system.find("e:TimeCreated", ns)
        if time_node is not None:
            timestamp = time_node.attrib.get("SystemTime", "")

    protocol = data.get("protocol", "tcp")
    src_ip = data.get("sourceip", "")
    src_port = safe_int(data.get("sourceport", 0))
    dst_ip = data.get("destinationip", "")
    dst_port = safe_int(data.get("destinationport", 0))

    return {
        "protocol": protocol,
        "src_ip": src_ip,
        "src_port": src_port,
        "dst_ip": dst_ip,
        "dst_port": dst_port,
        "timestamp": timestamp,
    }


def run_live(channel: str, event_id: int, output_path: Path, threshold: Optional[float], start_at_end: bool) -> None:
    try:
        import win32evtlog
    except Exception as exc:
        raise SystemExit(
            "pywin32 is required for live Windows logs. Install with: pip install pywin32"
        ) from exc

    logger = setup_logger("live_windows", LIVE_LOG_PATH)
    model, model_name, model_threshold = _load_model()
    threshold = model_threshold if threshold is None else threshold

    logger.info("Starting Windows live IDS on channel=%s event_id=%d", channel, event_id)
    logger.info("Model: %s | Threshold: %.3f", model_name, threshold)

    expected_cols = expected_feature_columns(COLUMNS, DROP_COLS, TARGET_COL)
    recent = deque(maxlen=WINDOW_SIZE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()

    try:
        query = f"*[System/EventID={event_id}]"
        flags = win32evtlog.EvtSubscribeToFutureEvents if start_at_end else win32evtlog.EvtSubscribeStartAtOldestRecord
        subscription = win32evtlog.EvtSubscribe(channel, flags, None, query)
    except Exception as exc:
        raise SystemExit(f"Failed to subscribe to event log: {exc}") from exc

    with output_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["timestamp", "src", "dst", "service", "score", "prediction"],
        )
        if not file_exists:
            writer.writeheader()

        while True:
            try:
                events = win32evtlog.EvtNext(subscription, 10)
            except Exception:
                events = []

            if not events:
                time.sleep(POLL_SECONDS)
                continue

            for event in events:
                try:
                    xml = win32evtlog.EvtRender(event, win32evtlog.EvtRenderEventXml)
                except Exception:
                    continue

                data = _parse_sysmon_xml(xml)
                if data is None:
                    continue

                service = _map_service(data["dst_port"])
                feature_row = compute_flow_features(
                    src=data["src_ip"],
                    dst=data["dst_ip"],
                    service=service,
                    proto=str(data["protocol"]).lower(),
                    flag="SF",
                    src_port=data["src_port"],
                    duration=0.0,
                    src_bytes=0,
                    dst_bytes=0,
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
                        "timestamp": data["timestamp"],
                        "src": data["src_ip"],
                        "dst": data["dst_ip"],
                        "service": service,
                        "score": round(score, 4),
                        "prediction": "attack" if pred == 1 else "normal",
                    }
                )
                csvfile.flush()

                if pred == 1:
                    logger.warning("ALERT src=%s dst=%s service=%s score=%.3f", data["src_ip"], data["dst_ip"], service, score)
                else:
                    logger.info("OK src=%s dst=%s service=%s score=%.3f", data["src_ip"], data["dst_ip"], service, score)


def main():
    parser = argparse.ArgumentParser(description="Live IDS from Windows Event Log (Sysmon Event ID 3)")
    parser.add_argument(
        "--channel",
        default="Microsoft-Windows-Sysmon/Operational",
        help="Windows Event Log channel",
    )
    parser.add_argument(
        "--event-id",
        type=int,
        default=3,
        help="Event ID to monitor (Sysmon Network Connection is 3)",
    )
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
        "--start-at-end",
        action="store_true",
        help="Ignore old events and only watch new ones",
    )

    args = parser.parse_args()
    run_live(args.channel, args.event_id, Path(args.output), args.threshold, args.start_at_end)


if __name__ == "__main__":
    main()
