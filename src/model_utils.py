from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def get_models(random_state: int) -> Dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=1,
            class_weight="balanced",
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state,
            class_weight="balanced",
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Linear SVM": LinearSVC(
            max_iter=2000,
            class_weight="balanced",
        ),
    }


def get_param_grids() -> Dict[str, Dict[str, list]]:
    return {
        "Logistic Regression": {
            "model__C": [0.1, 1.0, 10.0],
        },
        "Random Forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 20],
        },
        "Decision Tree": {
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
        },
        "K-Nearest Neighbors": {
            "model__n_neighbors": [3, 5, 7],
        },
        "Linear SVM": {
            "model__C": [0.5, 1.0, 2.0],
        },
    }


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def compute_confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)

