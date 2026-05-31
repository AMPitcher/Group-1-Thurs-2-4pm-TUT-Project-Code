from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from preprocessing import SEED


def get_base_model():
    """Default KNN with no tuning."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_jobs=-1)),
        ]
    )


def get_model():
    """Tuned weighted KNN."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=15,
                    weights="distance",
                    metric="manhattan",
                    n_jobs=-1,
                ),
            ),
        ]
    )
