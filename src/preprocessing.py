from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import inspect

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_COLS, DROP_COLS, TARGET_COL


def _binarize_labels(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: 0 if str(x).strip() == "normal" else 1)


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.drop(columns=DROP_COLS, errors="ignore")
    y = _binarize_labels(df[TARGET_COL])
    X = df.drop(columns=[TARGET_COL])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    onehot_kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        onehot_kwargs["sparse_output"] = False
    else:
        onehot_kwargs["sparse"] = False

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(**onehot_kwargs)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor

