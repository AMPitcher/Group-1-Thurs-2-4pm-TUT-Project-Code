from sklearn.ensemble import RandomForestClassifier
from preprocessing import SEED


def get_base_model():
    """Default Random Forest with no tuning."""
    return RandomForestClassifier(
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def get_model():
    """Tuned Random Forest."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
