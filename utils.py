import joblib
import pandas as pd

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

model = joblib.load("models/churn_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")


# ---------------------------------------------------
# Create Input Data
# ---------------------------------------------------

def create_input(
    gender,
    senior,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges
):

    df = pd.DataFrame(columns=feature_names)

    df.loc[0] = 0

    # Numerical Features

    df["SeniorCitizen"] = senior
    df["tenure"] = tenure
    df["MonthlyCharges"] = monthly_charges
    df["TotalCharges"] = total_charges

    # Binary Features

    if "gender_Male" in df.columns:
        df["gender_Male"] = 1 if gender == "Male" else 0

    if "Partner_Yes" in df.columns:
        df["Partner_Yes"] = 1 if partner == "Yes" else 0

    if "Dependents_Yes" in df.columns:
        df["Dependents_Yes"] = 1 if dependents == "Yes" else 0

    if "PhoneService_Yes" in df.columns:
        df["PhoneService_Yes"] = 1 if phone_service == "Yes" else 0

    if "MultipleLines_Yes" in df.columns:
        df["MultipleLines_Yes"] = 1 if multiple_lines == "Yes" else 0

    if "MultipleLines_No phone service" in df.columns:
        df["MultipleLines_No phone service"] = (
            1 if multiple_lines == "No phone service" else 0
        )

    # Internet

    if "InternetService_Fiber optic" in df.columns:
        df["InternetService_Fiber optic"] = (
            1 if internet_service == "Fiber optic" else 0
        )

    if "InternetService_No" in df.columns:
        df["InternetService_No"] = (
            1 if internet_service == "No" else 0
        )

    # Online Security

    if "OnlineSecurity_Yes" in df.columns:
        df["OnlineSecurity_Yes"] = (
            1 if online_security == "Yes" else 0
        )

    if "OnlineSecurity_No internet service" in df.columns:
        df["OnlineSecurity_No internet service"] = (
            1 if online_security == "No internet service" else 0
        )

    # Online Backup

    if "OnlineBackup_Yes" in df.columns:
        df["OnlineBackup_Yes"] = (
            1 if online_backup == "Yes" else 0
        )

    if "OnlineBackup_No internet service" in df.columns:
        df["OnlineBackup_No internet service"] = (
            1 if online_backup == "No internet service" else 0
        )

    # Device Protection

    if "DeviceProtection_Yes" in df.columns:
        df["DeviceProtection_Yes"] = (
            1 if device_protection == "Yes" else 0
        )

    if "DeviceProtection_No internet service" in df.columns:
        df["DeviceProtection_No internet service"] = (
            1 if device_protection == "No internet service" else 0
        )

    # Tech Support

    if "TechSupport_Yes" in df.columns:
        df["TechSupport_Yes"] = (
            1 if tech_support == "Yes" else 0
        )

    if "TechSupport_No internet service" in df.columns:
        df["TechSupport_No internet service"] = (
            1 if tech_support == "No internet service" else 0
        )

    # Streaming TV

    if "StreamingTV_Yes" in df.columns:
        df["StreamingTV_Yes"] = (
            1 if streaming_tv == "Yes" else 0
        )

    if "StreamingTV_No internet service" in df.columns:
        df["StreamingTV_No internet service"] = (
            1 if streaming_tv == "No internet service" else 0
        )

    # Streaming Movies

    if "StreamingMovies_Yes" in df.columns:
        df["StreamingMovies_Yes"] = (
            1 if streaming_movies == "Yes" else 0
        )

    if "StreamingMovies_No internet service" in df.columns:
        df["StreamingMovies_No internet service"] = (
            1 if streaming_movies == "No internet service" else 0
        )

    # Contract

    if "Contract_One year" in df.columns:
        df["Contract_One year"] = (
            1 if contract == "One year" else 0
        )

    if "Contract_Two year" in df.columns:
        df["Contract_Two year"] = (
            1 if contract == "Two year" else 0
        )

    # Paperless Billing

    if "PaperlessBilling_Yes" in df.columns:
        df["PaperlessBilling_Yes"] = (
            1 if paperless_billing == "Yes" else 0
        )

    # Payment Method

    methods = [
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]

    for method in methods:

        column = f"PaymentMethod_{method}"

        if column in df.columns:

            df[column] = 1 if payment_method == method else 0

    return df


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

def predict(df):

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return prediction, probability


# ---------------------------------------------------
# Risk Level
# ---------------------------------------------------

def risk_level(probability):

    if probability < 0.30:
        return "🟢 Low Risk"

    elif probability < 0.70:
        return "🟡 Medium Risk"

    else:
        return "🔴 High Risk"


# ---------------------------------------------------
# Recommendation
# ---------------------------------------------------

def recommendation(probability):

    if probability < 0.30:

        return [
            "Customer is loyal.",
            "Continue regular engagement.",
            "Offer loyalty rewards."
        ]

    elif probability < 0.70:

        return [
            "Send promotional offers.",
            "Contact customer support.",
            "Provide personalized discounts."
        ]

    else:

        return [
            "Immediate retention campaign.",
            "Offer contract upgrade.",
            "Assign customer success manager.",
            "Provide premium support."
        ]