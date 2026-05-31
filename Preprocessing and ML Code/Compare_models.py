import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.utils.class_weight import compute_sample_weight

from preprocessing import load_and_prepare, SEED
import model_rf
import model_xgb
import model_svm
import model_knn
import model_mlp

CLASS_NAMES = ["poor", "moderate", "good"]
COLORS = ["#1D9E75", "#378ADD", "#D85A30", "#BA7517", "#993556"]

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "axes.grid.axis": "y",
    }
)


# ////////////////////////////////////////////////////////////
# ////// Load data //////
# ////////////////////////////////////////////////////////////

X_train, X_test, y_train, y_test = load_and_prepare()

# Full X/y for cross-validation (recombine train + test)
import pandas as pd

X_full = pd.concat([X_train, X_test])
y_full = np.concatenate([y_train, y_test])


# ////////////////////////////////////////////////////////////
# ////// Helper functions //////
# ////////////////////////////////////////////////////////////


def overfitting_report(name, model, X_train, y_train, X_test, y_test):
    train_f1 = f1_score(y_train, model.predict(X_train), average="macro")
    test_f1 = f1_score(y_test, model.predict(X_test), average="macro")
    print(f"\n=== {name} — Overfitting Check ===")
    print(f"Train macro F1: {train_f1:.3f}")
    print(f"Test  macro F1: {test_f1:.3f}")
    print(f"Gap:            {train_f1 - test_f1:.3f}")
    return train_f1, test_f1


def get_probabilities(model, X_test, y_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)
    else:
        cal = CalibratedClassifierCV(model, cv=5)
        cal.fit(X_train, y_train)
        return cal.predict_proba(X_test)


def run_cv(name, model, fit_params=None, n_splits=5):
    """Run stratified k-fold CV and return per-fold scores."""
    print(f"  Running {n_splits}-fold CV for {name}...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # CV doesn't support fit_params for sample_weight cleanly across all models,
    # so we score manually fold by fold for XGBoost, use cross_validate for others.
    if fit_params:
        fold_f1s, fold_acc, fold_roc = [], [], []
        for train_idx, val_idx in cv.split(X_full, y_full):
            Xtr, Xval = X_full.iloc[train_idx], X_full.iloc[val_idx]
            ytr, yval = y_full[train_idx], y_full[val_idx]
            sw = compute_sample_weight("balanced", y=ytr)
            m = model.__class__(**model.get_params())
            m.fit(Xtr, ytr, sample_weight=sw)
            ypred = m.predict(Xval)
            fold_f1s.append(f1_score(yval, ypred, average="macro"))
            fold_acc.append(accuracy_score(yval, ypred))
            try:
                yb = label_binarize(yval, classes=[0, 1, 2])
                yp = m.predict_proba(Xval)
                fold_roc.append(
                    roc_auc_score(yb, yp, multi_class="ovr", average="macro")
                )
            except Exception:
                fold_roc.append(np.nan)
        return np.array(fold_f1s), np.array(fold_acc), np.array(fold_roc)
    else:
        cv_res = cross_validate(
            model,
            X_full,
            y_full,
            cv=cv,
            scoring=["f1_macro", "accuracy"],
            n_jobs=-1,
        )
        return (
            cv_res["test_f1_macro"],
            cv_res["test_accuracy"],
            np.full(n_splits, np.nan),
        )


# ////////////////////////////////////////////////////////////
# ////// Define base and tuned model pairs //////
# ////////////////////////////////////////////////////////////

# Each entry: (display_name, base_model, tuned_model, fit_params_for_tuned)
model_pairs = [
    (
        "Random Forest",
        model_rf.get_base_model(),
        model_rf.get_model(),
        {},
    ),
    (
        "XGBoost",
        model_xgb.get_base_model(),
        model_xgb.get_model(),
        model_xgb.get_fit_params(y_train),
    ),
    (
        "SVM (LinearSVC)",
        model_svm.get_base_model(),
        model_svm.get_model(),
        {},
    ),
    (
        "KNN",
        model_knn.get_base_model(),
        model_knn.get_model(),
        {},
    ),
    (
        "MLP (Neural Net)",
        model_mlp.get_base_model(),
        model_mlp.get_model(),
        {},
    ),
]


# ////////////////////////////////////////////////////////////
# ////// Train, evaluate, collect results //////
# ////////////////////////////////////////////////////////////

results = []  # tuned model full results
before_after = []  # before/after tuning comparison
cv_scores = {}  # name -> (fold_f1s, fold_accs, fold_rocs)

for name, base_model, tuned_model, fit_params in model_pairs:
    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print("=" * 55)

    # --- Fit base model ---
    print("  Fitting base model...")
    base_model.fit(X_train, y_train)
    y_pred_base = base_model.predict(X_test)
    base_f1 = f1_score(y_test, y_pred_base, average="macro")
    base_acc = accuracy_score(y_test, y_pred_base)
    base_recall = recall_score(y_test, y_pred_base, labels=[0], average=None)[0]

    # --- Fit tuned model ---
    print("  Fitting tuned model...")
    tuned_model.fit(X_train, y_train, **fit_params)
    y_pred_tuned = tuned_model.predict(X_test)
    tuned_f1 = f1_score(y_test, y_pred_tuned, average="macro")
    tuned_acc = accuracy_score(y_test, y_pred_tuned)
    tuned_recall = recall_score(y_test, y_pred_tuned, labels=[0], average=None)[0]

    print(f"\n  Classification report (tuned):")
    print(classification_report(y_test, y_pred_tuned, target_names=CLASS_NAMES))

    train_f1, test_f1 = overfitting_report(
        name, tuned_model, X_train, y_train, X_test, y_test
    )

    # ROC-AUC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_proba = get_probabilities(tuned_model, X_test, y_test)
    roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")
    print(f"  ROC-AUC (macro OvR): {roc_auc:.3f}")

    # Store tuned results
    results.append(
        (name, tuned_model, y_pred_tuned, train_f1, test_f1, roc_auc, y_proba)
    )

    # Before/after tuning
    before_after.append(
        {
            "Model": name,
            "Base F1": round(base_f1, 3),
            "Tuned F1": round(tuned_f1, 3),
            "F1 change": round(tuned_f1 - base_f1, 3),
            "Base accuracy": round(base_acc, 3),
            "Tuned accuracy": round(tuned_acc, 3),
            "Accuracy change": round(tuned_acc - base_acc, 3),
            "Base poor recall": round(base_recall, 3),
            "Tuned poor recall": round(tuned_recall, 3),
            "Poor recall change": round(tuned_recall - base_recall, 3),
        }
    )

    # Confusion matrix — tuned
    cm = confusion_matrix(y_test, y_pred_tuned)
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(cmap="Blues")
    plt.title(f"{name} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        f"cm_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png",
        dpi=150,
    )
    plt.close()

    # Cross-validation on tuned model
    fold_f1s, fold_accs, fold_rocs = run_cv(
        name, tuned_model, fit_params if fit_params else None
    )
    cv_scores[name] = (fold_f1s, fold_accs, fold_rocs)
    print(f"  CV Macro F1:  {fold_f1s.mean():.3f} ± {fold_f1s.std():.3f}")
    print(f"  CV Accuracy:  {fold_accs.mean():.3f} ± {fold_accs.std():.3f}")


# ////////////////////////////////////////////////////////////
# ////// Before/after tuning table //////
# ////////////////////////////////////////////////////////////

print("\n" + "=" * 70)
print("BEFORE vs AFTER HYPERPARAMETER TUNING")
print("=" * 70)
ba_df = pd.DataFrame(before_after)
print(ba_df.to_string(index=False))
ba_df.to_csv("before_after_tuning.csv", index=False)
print("Saved: before_after_tuning.csv")


# ////////////////////////////////////////////////////////////
# ////// Plot 1: Before/after tuning bar chart //////
# ////////////////////////////////////////////////////////////

names = [r["Model"] for r in before_after]
base_f = [r["Base F1"] for r in before_after]
tune_f = [r["Tuned F1"] for r in before_after]

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(
    x - width / 2,
    base_f,
    width,
    label="Base model",
    color="#888780",
    alpha=0.85,
    zorder=3,
)
b2 = ax.bar(
    x + width / 2,
    tune_f,
    width,
    label="Tuned model",
    color="#1D9E75",
    alpha=0.85,
    zorder=3,
)

for bar in b1 + b2:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{bar.get_height():.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.set_ylabel("Macro F1 Score")
ax.set_title("Before vs After Hyperparameter Tuning — Macro F1")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha="right")
ax.set_ylim(0, 1.0)
ax.legend()
plt.tight_layout()
plt.savefig("plot_before_after_tuning.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot_before_after_tuning.png")

# ////////////////////////////////////////////////////////////
# ////// Plot: Poor recall before vs after tuning //////
# ////////////////////////////////////////////////////////////

poor_recall_base = [r["Base poor recall"] for r in before_after]
poor_recall_tuned = [r["Tuned poor recall"] for r in before_after]

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(
    x - width / 2,
    poor_recall_base,
    width,
    label="Base model",
    color="#888780",
    alpha=0.85,
    zorder=3,
)
b2 = ax.bar(
    x + width / 2,
    poor_recall_tuned,
    width,
    label="Tuned model",
    color="#D85A30",
    alpha=0.85,
    zorder=3,
)

for bar in list(b1) + list(b2):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{bar.get_height():.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.set_ylabel("Recall — Poor Class")
ax.set_title("Before vs After Hyperparameter Tuning — Poor Class Recall")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha="right")
ax.set_ylim(0, 1.0)
ax.legend()
plt.tight_layout()
plt.savefig("plot_before_after_poor_recall.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot_before_after_poor_recall.png")

# ////////////////////////////////////////////////////////////
# ////// Plot 2: CV boxplot — Macro F1 //////
# ////////////////////////////////////////////////////////////

fig, ax = plt.subplots(figsize=(10, 6))
cv_f1_data = [cv_scores[n][0] for n in names]

bp = ax.boxplot(
    cv_f1_data,
    labels=names,
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markersize=5, alpha=0.5),
)

for patch, color in zip(bp["boxes"], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

# Overlay individual fold points
for i, (fold_scores, color) in enumerate(zip(cv_f1_data, COLORS), start=1):
    jitter = np.random.RandomState(42).uniform(-0.1, 0.1, len(fold_scores))
    ax.scatter(
        [i + j for j in jitter],
        fold_scores,
        color=color,
        s=40,
        zorder=5,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.5,
    )

ax.set_ylabel("Macro F1 Score (CV folds)")
ax.set_title("Model Stability — Cross-Validation Macro F1 (5-fold)")
ax.set_xticklabels(names, rotation=15, ha="right")
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig("plot_cv_boxplot_f1.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot_cv_boxplot_f1.png")


# ////////////////////////////////////////////////////////////
# ////// Plot 3: CV boxplot — Accuracy //////
# ////////////////////////////////////////////////////////////

fig, ax = plt.subplots(figsize=(10, 6))
cv_acc_data = [cv_scores[n][1] for n in names]

bp = ax.boxplot(
    cv_acc_data,
    labels=names,
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markersize=5, alpha=0.5),
)

for patch, color in zip(bp["boxes"], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

for i, (fold_scores, color) in enumerate(zip(cv_acc_data, COLORS), start=1):
    jitter = np.random.RandomState(42).uniform(-0.1, 0.1, len(fold_scores))
    ax.scatter(
        [i + j for j in jitter],
        fold_scores,
        color=color,
        s=40,
        zorder=5,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.5,
    )

ax.set_ylabel("Accuracy (CV folds)")
ax.set_title("Model Stability — Cross-Validation Accuracy (5-fold)")
ax.set_xticklabels(names, rotation=15, ha="right")
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig("plot_cv_boxplot_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot_cv_boxplot_accuracy.png")


# ////////////////////////////////////////////////////////////
# ////// Plot 4: Holdout vs CV mean comparison //////
# ////////////////////////////////////////////////////////////

holdout_f1s = [r[4] for r in results]
cv_mean_f1s = [cv_scores[n][0].mean() for n in names]
cv_std_f1s = [cv_scores[n][0].std() for n in names]

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(
    x - width / 2,
    holdout_f1s,
    width,
    label="Holdout test F1",
    color="#378ADD",
    alpha=0.85,
    zorder=3,
)
b2 = ax.bar(
    x + width / 2,
    cv_mean_f1s,
    width,
    label="CV mean F1",
    color="#1D9E75",
    alpha=0.85,
    zorder=3,
    yerr=cv_std_f1s,
    capsize=5,
    error_kw=dict(elinewidth=1.2, ecolor="#0F6E56"),
)

for bar in list(b1) + list(b2):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.012,
        f"{bar.get_height():.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.set_ylabel("Macro F1 Score")
ax.set_title("Holdout Test F1 vs Cross-Validation Mean F1 (± std)")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha="right")
ax.set_ylim(0, 1.0)
ax.legend()
plt.tight_layout()
plt.savefig("plot_holdout_vs_cv.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot_holdout_vs_cv.png")


# ////////////////////////////////////////////////////////////
# ////// Plot 5: Train vs Test macro F1 //////
# ////////////////////////////////////////////////////////////

train_f1s = [r[3] for r in results]
test_f1s = [r[4] for r in results]

x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(
    x - width / 2,
    train_f1s,
    width,
    label="Train macro F1",
    color="#378ADD",
    alpha=0.85,
    zorder=3,
)
b2 = ax.bar(
    x + width / 2,
    test_f1s,
    width,
    label="Test macro F1",
    color="#1D9E75",
    alpha=0.85,
    zorder=3,
)

for bar in list(b1) + list(b2):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{bar.get_height():.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.set_ylabel("Macro F1 Score")
ax.set_title("Model Comparison — Train vs Test Macro F1 (Overfitting Check)")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha="right")
ax.set_ylim(0, 1.05)
ax.legend()
plt.tight_layout()
plt.savefig("comparison_f1_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: comparison_f1_bar.png")

# ////////////////////////////////////////////////////////////
# ////// Plot: Overfitting gap before and after tuning //////
# ////////////////////////////////////////////////////////////

print("\nCalculating overfitting gaps...")

gap_data = []

for name, base_model, tuned_model, fit_params in model_pairs:
    # Base model gap
    base_train_f1 = f1_score(y_train, base_model.predict(X_train), average="macro")
    base_test_f1 = f1_score(y_test, base_model.predict(X_test), average="macro")
    base_gap = base_train_f1 - base_test_f1

    # Tuned model gap
    tuned_train_f1 = f1_score(y_train, tuned_model.predict(X_train), average="macro")
    tuned_test_f1 = f1_score(y_test, tuned_model.predict(X_test), average="macro")
    tuned_gap = tuned_train_f1 - tuned_test_f1

    gap_data.append(
        {
            "Model": name,
            "Base gap": round(base_gap, 3),
            "Tuned gap": round(tuned_gap, 3),
        }
    )

    print(f"{name:20s} | Base gap: {base_gap:.3f}  →  Tuned gap: {tuned_gap:.3f}")

gap_df = pd.DataFrame(gap_data)
names_gap = gap_df["Model"].tolist()
base_gaps = gap_df["Base gap"].tolist()
tuned_gaps = gap_df["Tuned gap"].tolist()

x = np.arange(len(names_gap))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(
    x - width / 2,
    base_gaps,
    width,
    label="Base model gap",
    color="#888780",
    alpha=0.85,
    zorder=3,
)
b2 = ax.bar(
    x + width / 2,
    tuned_gaps,
    width,
    label="Tuned model gap",
    color="#D85A30",
    alpha=0.85,
    zorder=3,
)

# Value labels
for bar in list(b1) + list(b2):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{bar.get_height():.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# Reference line for healthy gap threshold
ax.axhline(
    y=0.05,
    color="#E24B4A",
    linestyle="--",
    linewidth=1.2,
    alpha=0.7,
    label="Mild overfitting threshold (0.05)",
)
ax.axhline(
    y=0.15,
    color="#A32D2D",
    linestyle="--",
    linewidth=1.2,
    alpha=0.7,
    label="Significant overfitting threshold (0.15)",
)

ax.set_ylabel("Train F1 − Test F1 (gap)")
ax.set_title("Overfitting Gap — Before vs After Hyperparameter Tuning")
ax.set_xticks(x)
ax.set_xticklabels(names_gap, rotation=15, ha="right")
ax.set_ylim(0, max(base_gaps + tuned_gaps) * 1.25)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("plot_overfitting_gap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot_overfitting_gap.png")


# ////////////////////////////////////////////////////////////
# ////// Plot 6: ROC-AUC bar chart //////
# ////////////////////////////////////////////////////////////

roc_aucs = [r[5] for r in results]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(names, roc_aucs, color=COLORS, alpha=0.85, zorder=3)

for bar, val in zip(bars, roc_aucs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

ax.set_ylabel("ROC-AUC (macro OvR)")
ax.set_title("Model Comparison — ROC-AUC Score")
ax.set_ylim(0, 1.05)
ax.set_xticklabels(names, rotation=15, ha="right")
plt.tight_layout()
plt.savefig("comparison_roc_auc_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: comparison_roc_auc_bar.png")


# ////////////////////////////////////////////////////////////
# ////// Plot 7: ROC curves per class //////
# ////////////////////////////////////////////////////////////

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle("ROC Curves — One-vs-Rest per Class", fontsize=13)

for class_idx, class_name in enumerate(CLASS_NAMES):
    ax = axes[class_idx]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    for i, (name, _, _, _, _, _, y_proba) in enumerate(results):
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_proba[:, class_idx])
        roc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[i], label=f"{name} (AUC={roc_val:.2f})", lw=1.8)
    ax.set_title(f"Class: {class_name}")
    ax.set_xlabel("False Positive Rate")
    if class_idx == 0:
        ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("comparison_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: comparison_roc_curves.png")


# ////////////////////////////////////////////////////////////
# ////// CV summary table //////
# ////////////////////////////////////////////////////////////

print("\n" + "=" * 70)
print("CROSS-VALIDATION SUMMARY (5-fold, tuned models)")
print("=" * 70)

cv_summary_rows = []
for name in names:
    f1s, accs, rocs = cv_scores[name]
    holdout_f1 = next(r[4] for r in results if r[0] == name)
    cv_summary_rows.append(
        {
            "Model": name,
            "Holdout F1": round(holdout_f1, 3),
            "CV mean F1": round(f1s.mean(), 3),
            "CV std F1": round(f1s.std(), 3),
            "Holdout - CV F1": round(holdout_f1 - f1s.mean(), 3),
            "CV mean accuracy": round(accs.mean(), 3),
            "CV std accuracy": round(accs.std(), 3),
        }
    )

cv_summary_df = pd.DataFrame(cv_summary_rows)
print(cv_summary_df.to_string(index=False))
cv_summary_df.to_csv("cv_summary.csv", index=False)
print("\nSaved: cv_summary.csv")

print("\n" + "=" * 70)
print("All outputs saved.")
print("=" * 70)
print("""
Files produced:
  before_after_tuning.csv
  cv_summary.csv
  plot_before_after_tuning.png
  plot_cv_boxplot_f1.png
  plot_cv_boxplot_accuracy.png
  plot_holdout_vs_cv.png
  comparison_f1_bar.png
  comparison_roc_auc_bar.png
  comparison_roc_curves.png
  cm_<model>.png  (one per model)
""")

# ////////////////////////////////////////////////////////////
# ////// Feature selection experiment — all models //////
# ////////////////////////////////////////////////////////////

print("\n" + "=" * 55)
print("FEATURE SELECTION EXPERIMENT — ALL MODELS")
print("=" * 55)

# Get top 40 feature list from RF importances (already computed above)
rf_model = next(r[1] for r in results if r[0] == "Random Forest")

importances = pd.Series(
    rf_model.feature_importances_, index=X_train.columns
).sort_values(ascending=False)  # this line was missing

top_40_features = importances.head(40).index.tolist()
X_train_40 = X_train[top_40_features]
X_test_40 = X_test[top_40_features]

selection_results = []

for name, base_model, tuned_model, fit_params in model_pairs:
    # Full 49 features — already trained, just re-score
    y_pred_full = tuned_model.predict(X_test)
    f1_full = f1_score(y_test, y_pred_full, average="macro")
    rec_full = recall_score(y_test, y_pred_full, labels=[0], average=None)[0]

    # Top 40 features — refit and score
    m40 = (
        model_rf.get_model()
        if name == "Random Forest"
        else model_xgb.get_model()
        if name == "XGBoost"
        else model_svm.get_model()
        if name == "SVM (LinearSVC)"
        else model_knn.get_model()
        if name == "KNN"
        else model_mlp.get_model()
    )

    m40.fit(X_train_40, y_train, **fit_params)
    y_pred_40 = m40.predict(X_test_40)
    f1_40 = f1_score(y_test, y_pred_40, average="macro")
    rec_40 = recall_score(y_test, y_pred_40, labels=[0], average=None)[0]

    selection_results.append(
        {
            "Model": name,
            "F1 (49 features)": round(f1_full, 3),
            "F1 (40 features)": round(f1_40, 3),
            "F1 change": round(f1_40 - f1_full, 3),
            "Recall (49)": round(rec_full, 3),
            "Recall (40)": round(rec_40, 3),
            "Recall change": round(rec_40 - rec_full, 3),
        }
    )

    print(
        f"{name:20s} | F1: {f1_full:.3f} → {f1_40:.3f} ({f1_40 - f1_full:+.3f}) | "
        f"Poor recall: {rec_full:.3f} → {rec_40:.3f} ({rec_40 - rec_full:+.3f})"
    )

sel_df = pd.DataFrame(selection_results)
sel_df.to_csv("feature_selection_experiment.csv", index=False)
print("\nSaved: feature_selection_experiment.csv")

# ////////////////////////////////////////////////////////////
# ////// Plot: Top 30 feature importances (horizontal) //////
# ////////////////////////////////////////////////////////////

top_30 = importances.head(30).sort_values(
    ascending=True
)  # ascending=True puts highest at top

fig, ax = plt.subplots(figsize=(10, 10))
bars = ax.barh(top_30.index, top_30.values, color="#1D9E75", alpha=0.85)

ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
ax.set_title("Top 30 Feature Importances — Random Forest")
ax.tick_params(axis="y", labelsize=10)
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.grid(axis="y", visible=False)

plt.tight_layout()
plt.savefig("plot_feature_importance_top30.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot_feature_importance_top30.png")
