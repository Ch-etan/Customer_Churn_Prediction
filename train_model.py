import os
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from preprocess import load_data, prepare_data


# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ============================================================
# Create Models Folder
# ============================================================

os.makedirs("models", exist_ok=True)


# ============================================================
# Load Dataset
# ============================================================

print("=" * 60)
print("CUSTOMER CHURN PREDICTION")
print("=" * 60)

df = load_data()

print("\nDataset Shape:")
print(df.shape)

print("\nChurn Distribution:")
print(df["Churn"].value_counts())


# ============================================================
# Prepare Data
# ============================================================

X_train, X_test, y_train, y_test = prepare_data(df)


# ============================================================
# Identify Columns
# ============================================================

categorical_columns = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_columns = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nNumeric Columns:")
print(numeric_columns)

print("\nCategorical Columns:")
print(categorical_columns)


# ============================================================
# Preprocessing Pipeline
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            drop="first"
        )
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_columns
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_columns
    )
])


# ============================================================
# Models
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    )
}


if XGBOOST_AVAILABLE:

    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )


# ============================================================
# Train and Evaluate
# ============================================================

results = {}

trained_models = {}

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)


for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    results[name] = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4)
    }

    trained_models[name] = pipeline

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")


# ============================================================
# Select Best Model
# ============================================================

best_name = max(
    results,
    key=lambda name: results[name]["roc_auc"]
)

best_model = trained_models[best_name]


print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Best Model:", best_name)
print(
    "ROC-AUC:",
    results[best_name]["roc_auc"]
)


# ============================================================
# Save Best Model
# ============================================================

joblib.dump(
    best_model,
    "models/churn_model.pkl"
)


# ============================================================
# Save Metrics
# ============================================================

with open(
    "models/metrics.json",
    "w"
) as file:

    json.dump(
        {
            "best_model": best_name,
            "results": results,
            "dataset_rows": int(len(df)),
            "dataset_columns": int(len(df.columns))
        },
        file,
        indent=4
    )


# ============================================================
# Save Feature Names
# ============================================================

joblib.dump(
    X_train.columns.tolist(),
    "models/feature_names.pkl"
)


print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print("\nModel saved:")
print("models/churn_model.pkl")

print("\nMetrics saved:")
print("models/metrics.json")

print("\nFeature names saved:")
print("models/feature_names.pkl")

print("\nProject training completed successfully!")