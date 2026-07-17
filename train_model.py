# ============================================================
# Customer Churn Prediction
# Model Training
# ============================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# Load Dataset
# ============================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Original Shape :", df.shape)

# ============================================================
# Data Cleaning
# ============================================================

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df.dropna(inplace=True)

df.drop("customerID", axis=1, inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print("Shape After Cleaning :", df.shape)

# ============================================================
# One-Hot Encoding
# ============================================================

df = pd.get_dummies(df, drop_first=True)

# Convert Boolean Columns to Integer

bool_columns = df.select_dtypes(include=["bool"]).columns
df[bool_columns] = df[bool_columns].astype(int)

print("Shape After Encoding :", df.shape)

# ============================================================
# Features & Target
# ============================================================

X = df.drop("Churn", axis=1)
y = df["Churn"]

# ============================================================
# Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# ============================================================
# Logistic Regression
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_prob = lr_model.predict_proba(X_test)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_prob)

print(f"Accuracy  : {lr_accuracy:.4f}")
print(f"Precision : {lr_precision:.4f}")
print(f"Recall    : {lr_recall:.4f}")
print(f"F1 Score  : {lr_f1:.4f}")
print(f"ROC AUC   : {lr_auc:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, lr_pred))

print("\nClassification Report")
print(classification_report(y_test, lr_pred))

# ============================================================
# Random Forest
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)

print(f"Accuracy  : {rf_accuracy:.4f}")
print(f"Precision : {rf_precision:.4f}")
print(f"Recall    : {rf_recall:.4f}")
print(f"F1 Score  : {rf_f1:.4f}")
print(f"ROC AUC   : {rf_auc:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report")
print(classification_report(y_test, rf_pred))

# ============================================================
# Model Comparison
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        lr_accuracy,
        rf_accuracy
    ],
    "Precision": [
        lr_precision,
        rf_precision
    ],
    "Recall": [
        lr_recall,
        rf_recall
    ],
    "F1 Score": [
        lr_f1,
        rf_f1
    ],
    "ROC AUC": [
        lr_auc,
        rf_auc
    ]
})

print("\n")
print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(comparison)

# ============================================================
# Save Best Model
# ============================================================

if lr_accuracy >= rf_accuracy:
    best_model = lr_model
    best_name = "Logistic Regression"
else:
    best_model = rf_model
    best_name = "Random Forest"

joblib.dump(best_model, "models/churn_model.pkl")

# Save Feature Names for Streamlit
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

print("\n")
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Best Model : {best_name}")

print("\nModel Saved Successfully!")

print("Model File   : models/churn_model.pkl")
print("Feature File : models/feature_names.pkl")

print("\nProject Completed Successfully!")