from collections import deque
from typing import Deque, Dict

import pandas as pd

SERROR_STATES = {"S0", "S1", "S2", "S3", "S4"}
RERROR_STATES = {"REJ", "RSTR"}


def safe_float(value) -> float:
    try:
        if value in ("-", "", None):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def safe_int(value) -> int:
    try:
        if value in ("-", "", None):
            return 0
        return int(float(value))
    except Exception:
        return 0


def _rate(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_flow_features(
    *,
    src: str,
    dst: str,
    service: str,
    proto: str,
    flag: str,
    src_port: int,
    duration: float,
    src_bytes: int,
    dst_bytes: int,
    recent: Deque[Dict[str, object]],
) -> Dict[str, float]:
    land = 1 if src and dst and src == dst else 0

    serror = 1 if flag in SERROR_STATES else 0
    rerror = 1 if flag in RERROR_STATES else 0

    same_src = [r for r in recent if r["src"] == src]
    count = len(same_src)
    same_srv = [r for r in same_src if r["service"] == service]
    srv_count = len(same_srv)

    serror_rate = _rate(sum(r["serror"] for r in same_src), count)
    srv_serror_rate = _rate(sum(r["serror"] for r in same_srv), srv_count)
    rerror_rate = _rate(sum(r["rerror"] for r in same_src), count)
    srv_rerror_rate = _rate(sum(r["rerror"] for r in same_srv), srv_count)
    same_srv_rate = _rate(srv_count, count)
    diff_srv_rate = _rate(count - srv_count, count)
    srv_diff_host_rate = _rate(sum(1 for r in same_srv if r["dst"] != dst), srv_count)

    same_dst = [r for r in recent if r["dst"] == dst]
    dst_host_count = len(same_dst)
    dst_host_srv = [r for r in same_dst if r["service"] == service]
    dst_host_srv_count = len(dst_host_srv)
    dst_host_same_srv_rate = _rate(dst_host_srv_count, dst_host_count)
    dst_host_diff_srv_rate = _rate(dst_host_count - dst_host_srv_count, dst_host_count)
    dst_host_same_src_port_rate = _rate(
        sum(1 for r in same_dst if r["src_port"] == src_port),
        dst_host_count,
    )
    dst_host_srv_diff_host_rate = _rate(
        sum(1 for r in dst_host_srv if r["src"] != src),
        dst_host_srv_count,
    )
    dst_host_serror_rate = _rate(sum(r["serror"] for r in same_dst), dst_host_count)
    dst_host_srv_serror_rate = _rate(sum(r["serror"] for r in dst_host_srv), dst_host_srv_count)
    dst_host_rerror_rate = _rate(sum(r["rerror"] for r in same_dst), dst_host_count)
    dst_host_srv_rerror_rate = _rate(sum(r["rerror"] for r in dst_host_srv), dst_host_srv_count)

    features = {
        "duration": duration,
        "protocol_type": proto,
        "service": service,
        "flag": flag,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "land": land,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 0,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": count,
        "srv_count": srv_count,
        "serror_rate": serror_rate,
        "srv_serror_rate": srv_serror_rate,
        "rerror_rate": rerror_rate,
        "srv_rerror_rate": srv_rerror_rate,
        "same_srv_rate": same_srv_rate,
        "diff_srv_rate": diff_srv_rate,
        "srv_diff_host_rate": srv_diff_host_rate,
        "dst_host_count": dst_host_count,
        "dst_host_srv_count": dst_host_srv_count,
        "dst_host_same_srv_rate": dst_host_same_srv_rate,
        "dst_host_diff_srv_rate": dst_host_diff_srv_rate,
        "dst_host_same_src_port_rate": dst_host_same_src_port_rate,
        "dst_host_srv_diff_host_rate": dst_host_srv_diff_host_rate,
        "dst_host_serror_rate": dst_host_serror_rate,
        "dst_host_srv_serror_rate": dst_host_srv_serror_rate,
        "dst_host_rerror_rate": dst_host_rerror_rate,
        "dst_host_srv_rerror_rate": dst_host_srv_rerror_rate,
    }

    recent.append(
        {
            "src": src,
            "dst": dst,
            "service": service,
            "serror": serror,
            "rerror": rerror,
            "src_port": src_port,
        }
    )

    return features


def expected_feature_columns(columns, drop_cols, target_col):
    return [c for c in columns if c not in drop_cols + [target_col]]

