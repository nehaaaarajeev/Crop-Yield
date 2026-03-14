# ============================================================
# CROP YIELD ANALYSIS — STEP-BY-STEP MACHINE LEARNING PIPELINE
# ============================================================
# Dataset : Crop_Yield.csv
# Target  : Yield Success  (1 = Success, 0 = Failure)
# ============================================================

# ────────────────────────────────────────────────────────────
# STEP 1 — IMPORT PACKAGES
# ────────────────────────────────────────────────────────────
"""
EXPLANATION – STEP 1
We import all libraries needed for the full pipeline:
  • pandas / numpy      → data loading and numerical operations
  • matplotlib / seaborn → static plots (confusion matrix heatmaps,
                           feature-importance charts)
  • plotly               → interactive charts (used in Streamlit)
  • scikit-learn         → preprocessing, model training & evaluation
  • streamlit            → interactive web dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)

print("✅ Step 1 complete — all packages imported successfully.\n")

# ────────────────────────────────────────────────────────────
# STEP 2 — LOAD DATA & BASIC DATA CHECK
# ────────────────────────────────────────────────────────────
"""
EXPLANATION – STEP 2
We load the CSV and run a quick health-check to understand:
  • Shape       → number of rows and columns
  • Column names→ variable names
  • dtypes      → numeric vs. categorical columns
  • Null counts → how many missing values exist per column
  • describe()  → basic statistics (mean, std, min, max, quartiles)
  • Value counts for the target → class balance check
"""

df = pd.read_csv("Crop_Yield.csv", encoding="latin1")

print("── Dataset shape ──────────────────────────────────────")
print(f"  Rows : {df.shape[0]}   Columns : {df.shape[1]}\n")

print("── Column names & dtypes ──────────────────────────────")
print(df.dtypes.to_string(), "\n")

print("── Null value counts ──────────────────────────────────")
print(df.isnull().sum().to_string(), "\n")

print("── Descriptive statistics (numeric columns) ───────────")
print(df.describe().T.to_string(), "\n")

print("── Target class distribution ──────────────────────────")
print(df["Yield Success"].value_counts().to_string(), "\n")

print("✅ Step 2 complete — basic data check done.\n")

# ────────────────────────────────────────────────────────────
# STEP 3 — HANDLE NULL VALUES
# ────────────────────────────────────────────────────────────
"""
EXPLANATION – STEP 3
Although this dataset has no missing values, we write a
production-ready imputation strategy that is safe to run on
any version of the data:
  • Numeric columns   → fill with column mean
  • Categorical/object columns → fill with the mode (most
    frequent value in that column)
After imputation, we verify that no nulls remain.
"""

# Identify column types
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()

print(f"Numeric columns   : {numeric_cols}")
print(f"Categorical columns: {categorical_cols}\n")

# Impute numeric nulls with mean
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)
        print(f"  Filled '{col}' nulls with mean = {mean_val:.4f}")

# Impute categorical nulls with mode
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  Filled '{col}' nulls with mode = '{mode_val}'")

print(f"\nNull values remaining: {df.isnull().sum().sum()}")
print("✅ Step 3 complete — null handling done.\n")

# ────────────────────────────────────────────────────────────
# STEP 4 — LABEL ENCODING ON CATEGORICAL VARIABLES
# ────────────────────────────────────────────────────────────
"""
EXPLANATION – STEP 4
Tree-based models work with numeric inputs. We apply
LabelEncoder to every object/string column and:
  1. Save the integer-encoded value back into the dataframe.
  2. Record the mapping (original category → encoded integer)
     in a human-readable dictionary so results can always be
     interpreted back.
  3. Export the full mapping to 'label_encoding_mapping.csv'
     so analysts can audit or reverse-engineer predictions.

Note: 'Access to Credit' and 'Govt. Subsidy Received' are
already 0/1 integers and do NOT need encoding.
"""

df_encoded = df.copy()
le = LabelEncoder()
encoding_records = []

for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])
    for original, encoded in zip(le.classes_, le.transform(le.classes_)):
        encoding_records.append({
            "Column": col,
            "Original Value": original,
            "Encoded Value": int(encoded)
        })
    print(f"  Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Save mapping to CSV
mapping_df = pd.DataFrame(encoding_records)
mapping_df.to_csv("label_encoding_mapping.csv", index=False)
print("\nEncoding mapping saved to 'label_encoding_mapping.csv'")
print(mapping_df.to_string(index=False))
print("\n✅ Step 4 complete — label encoding done.\n")

# ────────────────────────────────────────────────────────────
# STEP 5 — SPLIT INTO FEATURES (X) AND LABEL (y)
# ────────────────────────────────────────────────────────────
"""
EXPLANATION – STEP 5
We separate the encoded dataframe into:
  • X → all columns except 'Yield Success' (the feature matrix)
  • y → the 'Yield Success' column (the target vector)
Printing shapes confirms the split is correct.
"""

LABEL_COL = "Yield Success"
X = df_encoded.drop(columns=[LABEL_COL])
y = df_encoded[LABEL_COL]

print(f"Feature matrix X : {X.shape}  →  {X.columns.tolist()}")
print(f"Target vector  y : {y.shape}  →  classes {y.unique().tolist()}")
print("\n✅ Step 5 complete — X / y split done.\n")

# ────────────────────────────────────────────────────────────
# STEP 6 — TRAIN / TEST SPLIT  (80:20, stratified)
# ────────────────────────────────────────────────────────────
"""
EXPLANATION – STEP 6
We split X and y into training and testing subsets:
  • test_size=0.20      → 20 % held out for evaluation
  • stratify=y          → class proportions are preserved in
                          both splits (important for imbalanced data)
  • random_state=42     → fixed seed for full reproducibility
"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print(f"Training set  : X_train {X_train.shape}  |  y_train {y_train.shape}")
print(f"Testing  set  : X_test  {X_test.shape}   |  y_test  {y_test.shape}")
print(f"Train class balance : {dict(y_train.value_counts().sort_index())}")
print(f"Test  class balance : {dict(y_test.value_counts().sort_index())}")
print("\n✅ Step 6 complete — train/test split done.\n")

# ────────────────────────────────────────────────────────────
# STEP 7 — TRAIN CLASSIFICATION MODELS
# ────────────────────────────────────────────────────────────
"""
EXPLANATION – STEP 7
We train three tree-based classification models one by one:
  1. Decision Tree   — simple, fully interpretable, prone to over-fitting
  2. Random Forest   — ensemble of many decision trees (bagging),
                       reduces variance and over-fitting
  3. Gradient Boosted Trees (GBT) — sequential ensemble that corrects
                       errors of previous trees (boosting), often the
                       best performer for tabular data

All three share the same random_state=42 for fair comparison.
"""

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosted Trees": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"  ✓ {name} trained.")

print("\n✅ Step 7 complete — all models trained.\n")

# ────────────────────────────────────────────────────────────
# STEP 8 — MODEL EVALUATION (TABULATED RESULTS)
# ────────────────────────────────────────────────────────────
"""
EXPLANATION – STEP 8
For every trained model we compute four key metrics:
  • Training Accuracy → how well the model fits the training data
                         (high value with large gap to test accuracy
                          signals over-fitting)
  • Testing  Accuracy → true out-of-sample performance
  • Precision         → of all predicted Successes, what fraction
                         were actually Successes?  (TP / (TP + FP))
  • Recall            → of all actual Successes, what fraction did
                         we correctly identify?    (TP / (TP + FN))

Results are printed in a clean tabular format.
"""

results = []
for name, model in trained_models.items():
    train_acc  = accuracy_score(y_train, model.predict(X_train))
    test_acc   = accuracy_score(y_test,  model.predict(X_test))
    precision  = precision_score(y_test, model.predict(X_test), zero_division=0)
    recall     = recall_score(y_test,    model.predict(X_test), zero_division=0)
    results.append({
        "Model"            : name,
        "Train Accuracy"   : round(train_acc, 4),
        "Test Accuracy"    : round(test_acc,  4),
        "Precision"        : round(precision, 4),
        "Recall"           : round(recall,    4)
    })

results_df = pd.DataFrame(results)
print("── Model Evaluation Results ───────────────────────────")
print(results_df.to_string(index=False))
print("\n✅ Step 8 complete — evaluation done.\n")

# ────────────────────────────────────────────────────────────
# STEP 9 — CONFUSION MATRICES (HEATMAPS)
# ────────────────────────────────────────────────────────────
"""
EXPLANATION – STEP 9
A confusion matrix shows four values for binary classification:
  • TP (True Positive)  → Predicted Success, Actually Success
  • TN (True Negative)  → Predicted Failure, Actually Failure
  • FP (False Positive) → Predicted Success, Actually Failure
  • FN (False Negative) → Predicted Failure, Actually Success

We plot each model's confusion matrix as a seaborn heatmap with
axis labels and annotations so every cell is unambiguous.
All three matrices are shown side by side in a single figure
and saved to 'confusion_matrices.png'.
"""

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Confusion Matrices — All Models", fontsize=16, fontweight="bold", y=1.02)

class_labels = ["Failure (0)", "Success (1)"]

for ax, (name, model) in zip(axes, trained_models.items()):
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        linewidths=0.5,
        linecolor="grey",
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_title(name, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Actual Label", fontsize=11)

    # Annotate quadrants
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            label = ""
            if i == 0 and j == 0: label = "TN"
            elif i == 0 and j == 1: label = "FP"
            elif i == 1 and j == 0: label = "FN"
            else: label = "TP"
            ax.text(j + 0.5, i + 0.75, label,
                    ha="center", va="center",
                    fontsize=10, color="darkorange", fontweight="bold")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("Confusion matrices saved to 'confusion_matrices.png'")
print("✅ Step 9 complete — confusion matrices plotted.\n")

# ────────────────────────────────────────────────────────────
# STEP 10 — FEATURE IMPORTANCE CHARTS
# ────────────────────────────────────────────────────────────
"""
EXPLANATION – STEP 10
All three models expose a `.feature_importances_` attribute
that quantifies how much each feature contributed to the
model's decisions (based on impurity reduction for Decision
Tree & Random Forest, and on loss reduction for GBT).

We plot a horizontal bar chart for each model, sorting features
from most to least important so the chart is easy to read.
All three charts are saved to 'feature_importance.png'.
"""

feature_names = X.columns.tolist()

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Feature Importance — All Tree-Based Models",
             fontsize=16, fontweight="bold", y=1.01)

colours = ["#2196F3", "#4CAF50", "#FF9800"]

for ax, (name, model), colour in zip(axes, trained_models.items(), colours):
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)
    sorted_feat = [feature_names[i] for i in sorted_idx]
    sorted_imp  = importances[sorted_idx]

    bars = ax.barh(sorted_feat, sorted_imp, color=colour, edgecolor="white", height=0.7)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8.5)
    ax.set_title(name, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_xlim(0, max(importances) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Feature importance charts saved to 'feature_importance.png'")
print("\n✅ Step 10 complete — feature importance plotted.")
print("\n🎉 Full analysis pipeline completed successfully!")
