from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from preprocessing import SEED


def get_base_model():
    """Default XGBoost with no tuning."""
    return XGBClassifier(
        random_state=SEED,
        eval_metric="mlogloss",
        n_jobs=-1,
    )


def get_model():
    """Tuned XGBoost."""
    return XGBClassifier(
        n_estimators=500,
        min_child_weight=20,
        max_depth=5,
        learning_rate=0.1,
        gamma=0.3,
        colsample_bytree=0.6,
        random_state=SEED,
        eval_metric="mlogloss",
        n_jobs=-1,
    )


def get_fit_params(y_train):
    """XGBoost needs sample weights passed at fit time."""
    return {"sample_weight": compute_sample_weight(class_weight="balanced", y=y_train)}
