from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from preprocessing import SEED


def get_base_model():
    """Default LinearSVC with no tuning."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def get_model():
    """Tuned LinearSVC."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                LinearSVC(
                    class_weight="balanced",
                    loss="squared_hinge",
                    C=np.float64(0.017433288221999882),
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )
