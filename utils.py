import joblib
import pandas as pd


MODEL_PATH = "models/churn_model.pkl"


def load_model():

    return joblib.load(MODEL_PATH)


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
    paperless,
    payment,
    monthly,
    total
):

    data = {

        "gender": gender,

        "SeniorCitizen": senior,

        "Partner": partner,

        "Dependents": dependents,

        "tenure": tenure,

        "PhoneService": phone_service,

        "MultipleLines": multiple_lines,

        "InternetService": internet_service,

        "OnlineSecurity": online_security,

        "OnlineBackup": online_backup,

        "DeviceProtection": device_protection,

        "TechSupport": tech_support,

        "StreamingTV": streaming_tv,

        "StreamingMovies": streaming_movies,

        "Contract": contract,

        "PaperlessBilling": paperless,

        "PaymentMethod": payment,

        "MonthlyCharges": monthly,

        "TotalCharges": total
    }

    return pd.DataFrame([data])


def predict(input_df):

    model = load_model()

    prediction = int(
        model.predict(input_df)[0]
    )

    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    return prediction, probability


def risk_level(probability):

    if probability < 0.30:
        return "LOW"

    elif probability < 0.70:
        return "MEDIUM"

    return "HIGH"


def recommendation(probability):

    if probability < 0.30:

        return [
            "Maintain regular customer communication.",
            "Offer loyalty rewards.",
            "Consider premium service recommendations."
        ]

    elif probability < 0.70:

        return [
            "Send personalized promotional offers.",
            "Improve customer engagement.",
            "Consider retention discounts."
        ]

    return [
        "Contact the customer immediately.",
        "Offer a retention discount.",
        "Consider contract upgrade options.",
        "Assign a customer retention specialist."
    ]