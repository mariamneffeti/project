"""
=============================================================
  Model Comparison & Evaluation
  Target: Policy_Cancelled_Post_Purchase (Binary Classification)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. LOAD CLEANED DATA
# ─────────────────────────────────────────────
df = pd.read_csv("train_cleaned.csv")

target_col = "Policy_Cancelled_Post_Purchase"
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"✅ Dataset loaded: {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"   Target distribution: {dict(y.value_counts())}")
print("=" * 70)


# ─────────────────────────────────────────────
# 1. DEFINE MODELS TO COMPARE
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=200, use_label_encoder=False,
                                         eval_metric="logloss", random_state=42,
                                         verbosity=0, n_jobs=-1),
    "LightGBM":            LGBMClassifier(n_estimators=200, random_state=42,
                                          verbose=-1, n_jobs=-1),
}


# ─────────────────────────────────────────────
# 2. CROSS-VALIDATED EVALUATION
# ─────────────────────────────────────────────
scoring = ["accuracy", "f1", "precision", "recall", "roc_auc"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
print("\n🔄 Running 5-Fold Stratified Cross-Validation ...\n")

for name, model in models.items():
    print(f"  ⏳ Training {name} ...")
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    results[name] = {
        "Accuracy":  cv_results["test_accuracy"].mean(),
        "F1 Score":  cv_results["test_f1"].mean(),
        "Precision": cv_results["test_precision"].mean(),
        "Recall":    cv_results["test_recall"].mean(),
        "ROC AUC":   cv_results["test_roc_auc"].mean(),
    }
    print(f"     ✅ {name}  →  Accuracy: {results[name]['Accuracy']:.4f}  |  "
          f"F1: {results[name]['F1 Score']:.4f}  |  AUC: {results[name]['ROC AUC']:.4f}")

print("\n" + "=" * 70)


# ─────────────────────────────────────────────
# 3. RESULTS SUMMARY TABLE
# ─────────────────────────────────────────────
results_df = pd.DataFrame(results).T.sort_values("ROC AUC", ascending=False)
results_df.index.name = "Model"

print("\n📊 MODEL COMPARISON RESULTS (sorted by ROC AUC)\n")
print(results_df.to_string(float_format="{:.4f}".format))
print()

# Highlight the best model
best_model_name = results_df["ROC AUC"].idxmax()
best_auc = results_df.loc[best_model_name, "ROC AUC"]
print(f"🏆 Best Model: {best_model_name} (ROC AUC = {best_auc:.4f})")
print("=" * 70)

# Save the table
results_df.to_csv("model_comparison_results.csv")
print("✅ Results saved to: model_comparison_results.csv")


# ─────────────────────────────────────────────
# 4. VISUALIZATION: Metric Comparison Bar Chart
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
fig.suptitle("Model Comparison — 5-Fold Cross-Validation", fontsize=16, fontweight="bold", y=1.02)

colors = sns.color_palette("viridis", n_colors=len(results_df))

for i, metric in enumerate(["Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC"]):
    ax = axes[i]
    bars = ax.barh(results_df.index, results_df[metric], color=colors, edgecolor="white")
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.axvline(x=results_df[metric].max(), color="red", linestyle="--", alpha=0.5)
    # Add value labels on bars
    for bar, val in zip(bars, results_df[metric]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("comparison_bar_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: comparison_bar_chart.png")


# ─────────────────────────────────────────────
# 5. VISUALIZATION: ROC Curves (all models)
# ─────────────────────────────────────────────
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

fig, ax = plt.subplots(figsize=(9, 7))
color_list = sns.color_palette("tab10", n_colors=len(models))

for (name, model), color in zip(models.items(), color_list):
    model.fit(X_train, y_train)
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax,
                                    name=name, alpha=0.8, color=color)

ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC=0.5)")
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: roc_curves.png")


# ─────────────────────────────────────────────
# 6. VISUALIZATION: Confusion Matrices (top 3)
# ─────────────────────────────────────────────
top3 = results_df.head(3).index.tolist()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("Confusion Matrices — Top 3 Models", fontsize=14, fontweight="bold")

for ax, name in zip(axes, top3):
    model = models[name]
    # model is already fitted from ROC section above
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Cancelled", "Cancelled"],
                yticklabels=["Not Cancelled", "Cancelled"])
    acc = accuracy_score(y_test, y_pred)
    ax.set_title(f"{name}\n(Acc: {acc:.3f})", fontsize=11)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: confusion_matrices.png")


# ─────────────────────────────────────────────
# 7. VISUALIZATION: Radar Chart (top 3 models)
# ─────────────────────────────────────────────
metrics = ["Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC"]
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
radar_colors = sns.color_palette("Set2", n_colors=3)

for i, name in enumerate(top3):
    values = results_df.loc[name, metrics].tolist()
    values += values[:1]
    ax.plot(angles, values, "o-", linewidth=2, label=name, color=radar_colors[i])
    ax.fill(angles, values, alpha=0.15, color=radar_colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylim(0, 1)
ax.set_title("Radar Chart — Top 3 Models", fontsize=14, fontweight="bold", pad=20)
ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0), fontsize=9)
plt.tight_layout()
plt.savefig("radar_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: radar_chart.png")


# ─────────────────────────────────────────────
# 8. DETAILED REPORT FOR BEST MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"📋 DETAILED CLASSIFICATION REPORT — {best_model_name}")
print("=" * 70)
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best,
      target_names=["Not Cancelled", "Cancelled"]))

f1_macro = f1_score(y_test, y_pred_best, average="macro")
print(f"🎯 F1 Macro Score ({best_model_name}): {f1_macro:.4f}")

# ─────────────────────────────────────────────
# 9. SAVE BEST MODEL TO model.pkl
# ─────────────────────────────────────────────
import joblib

# Train the best model on the FULL dataset for maximum performance
best_model_final = models[best_model_name]
best_model_final.fit(X, y)

joblib.dump(best_model_final, "model.pkl")
print(f"\n✅ Best model ({best_model_name}) saved to: model.pkl")

print("\n✅ All done! Check the generated files:")
print("   • model.pkl                 — trained best model (ready to load & predict)")
print("   • comparison_bar_chart.png  — side-by-side metric bars")
print("   • roc_curves.png            — ROC curves for all models")
print("   • confusion_matrices.png    — confusion matrices (top 3)")
print("   • radar_chart.png           — radar/spider chart (top 3)")
