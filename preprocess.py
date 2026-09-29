import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove rows containing missing values
    df.dropna(inplace=True)

    # Remove customer ID because it is not useful for prediction
    df.drop("customerID", axis=1, inplace=True)

    return df


def prepare_data(df):

    # Convert target variable
    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    df = load_data()

    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nChurn Distribution:")
    print(df["Churn"].value_counts())

    print("\nChurn Percentage:")
    print(df["Churn"].value_counts(normalize=True) * 100)

    print("\nPreprocessing completed successfully!")