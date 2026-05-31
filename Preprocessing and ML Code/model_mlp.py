from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from preprocessing import SEED


def get_base_model():
    """Default MLP with no tuning."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    max_iter=300,
                    random_state=SEED,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )


def get_model():
    """Tuned MLP with best params from RandomizedSearchCV."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    learning_rate_init=0.001,
                    alpha=np.float64(0.004641588833612777),
                    activation="tanh",
                    max_iter=300,
                    random_state=SEED,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )
