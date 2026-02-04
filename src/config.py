from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

NSL_KDD_TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
NSL_KDD_TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"

COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]

TARGET_COL = "label"
DROP_COLS = ["difficulty"]
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
ONNX_MODEL_PATH = MODELS_DIR / "best_model.onnx"
METRICS_PATH = REPORTS_DIR / "metrics.csv"
METRICS_PLOT_PATH = REPORTS_DIR / "metrics_comparison.png"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix_best.png"
FEATURE_IMPORTANCE_CSV = REPORTS_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_PLOT = REPORTS_DIR / "feature_importance.png"
SHAP_IMPORTANCE_CSV = REPORTS_DIR / "shap_importance.csv"
SHAP_SUMMARY_PLOT = REPORTS_DIR / "shap_summary.png"
CALIBRATION_METRICS_PATH = REPORTS_DIR / "calibration_metrics.json"
TRAIN_LOG_PATH = REPORTS_DIR / "train.log"
PREDICT_LOG_PATH = REPORTS_DIR / "predict.log"
LIVE_ALERTS_PATH = REPORTS_DIR / "live_alerts.csv"
LIVE_LOG_PATH = REPORTS_DIR / "live_ids.log"

