import urllib.request
from pathlib import Path

import pandas as pd

from .config import (
    COLUMNS,
    DATA_DIR,
    NSL_KDD_TEST_URL,
    NSL_KDD_TRAIN_URL,
)


TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    urllib.request.urlretrieve(url, dest)


def ensure_dataset() -> None:
    _download(NSL_KDD_TRAIN_URL, TRAIN_FILE)
    _download(NSL_KDD_TEST_URL, TEST_FILE)


def load_raw_dataset() -> pd.DataFrame:
    train_df, test_df = load_train_test()
    return pd.concat([train_df, test_df], ignore_index=True)


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dataset()
    train_df = pd.read_csv(
        TRAIN_FILE,
        names=COLUMNS,
        header=None,
        na_values=["?"],
    )
    test_df = pd.read_csv(
        TEST_FILE,
        names=COLUMNS,
        header=None,
        na_values=["?"],
    )
    return train_df, test_df

