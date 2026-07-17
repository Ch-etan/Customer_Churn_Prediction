# ==========================================
# Customer Churn Prediction
# Data Preprocessing
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------------------
# Load Dataset
# ------------------------------------------
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("=" * 50)
print("Dataset Loaded Successfully!")
print("=" * 50)

print("\nOriginal Dataset Shape:")
print(df.shape)

# ------------------------------------------
# Convert TotalCharges to Numeric
# ------------------------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# ------------------------------------------
# Remove Missing Values
# ------------------------------------------
df.dropna(inplace=True)

print("\nDataset Shape After Removing Missing Values:")
print(df.shape)

# ------------------------------------------
# Remove CustomerID
# ------------------------------------------
df.drop("customerID", axis=1, inplace=True)

# ------------------------------------------
# Convert Target Variable
# ------------------------------------------
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ------------------------------------------
# One-Hot Encoding
# ------------------------------------------
df = pd.get_dummies(df, drop_first=True)

# ------------------------------------------
# Convert Boolean Columns to Integer
# ------------------------------------------
bool_columns = df.select_dtypes(include=["bool"]).columns
df[bool_columns] = df[bool_columns].astype(int)

print("\nDataset Shape After Encoding:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

# ------------------------------------------
# Separate Features and Target
# ------------------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeature Matrix Shape:")
print(X.shape)

print("\nTarget Vector Shape:")
print(y.shape)

# ------------------------------------------
# Train-Test Split
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 50)
print("Train-Test Split Completed")
print("=" * 50)

print("\nTraining Features :", X_train.shape)
print("Testing Features  :", X_test.shape)
print("Training Labels   :", y_train.shape)
print("Testing Labels    :", y_test.shape)

print("\nPreprocessing Completed Successfully!")