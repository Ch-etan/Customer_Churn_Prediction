import json
import os

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils import (
    create_input,
    predict,
    risk_level,
    recommendation
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

if os.path.exists("style.css"):

    with open("style.css", "r") as file:
        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True
        )


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = (
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

METRICS_PATH = "models/metrics.json"


@st.cache_data
def load_dataset():

    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    return df


@st.cache_data
def load_metrics():

    with open(METRICS_PATH, "r") as file:
        return json.load(file)


df = load_dataset()
metrics = load_metrics()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        📊 Customer Churn Prediction
    </div>

    <div class="sub-title">
        AI-Powered Customer Retention Analytics Dashboard
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("---")


# ============================================================
# KPI CARDS
# ============================================================

total_customers = len(df)

churned = (
    df["Churn"]
    .value_counts()
    .get("Yes", 0)
)

churn_rate = (
    churned / total_customers * 100
)


best_model = metrics["best_model"]

best_accuracy = (
    metrics["results"]
    [best_model]
    ["accuracy"] * 100
)


k1, k2, k3, k4 = st.columns(4)


with k1:

    st.metric(
        "👥 Total Customers",
        f"{total_customers:,}"
    )


with k2:

    st.metric(
        "⚠ Churn Rate",
        f"{churn_rate:.2f}%"
    )


with k3:

    st.metric(
        "🎯 Accuracy",
        f"{best_accuracy:.2f}%"
    )


with k4:

    st.metric(
        "🤖 Best Model",
        best_model
    )


# ============================================================
# ANALYTICS
# ============================================================

st.markdown("---")

st.header("📈 Customer Analytics")


chart1, chart2 = st.columns(2)


with chart1:

    churn_counts = (
        df["Churn"]
        .value_counts()
        .rename(
            index={
                "No": "Retained",
                "Yes": "Churned"
            }
        )
    )

    fig = go.Figure(
        data=[
            go.Pie(
                labels=churn_counts.index,
                values=churn_counts.values,
                hole=0.55
            )
        ]
    )

    fig.update_layout(
        title="Customer Churn Distribution",
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with chart2:

    model_names = list(
        metrics["results"].keys()
    )

    model_accuracy = [
        metrics["results"][name]["accuracy"] * 100
        for name in model_names
    ]

    fig = px.bar(
        x=model_names,
        y=model_accuracy,
        text=[
            f"{value:.2f}%"
            for value in model_accuracy
        ],
        labels={
            "x": "Model",
            "y": "Accuracy (%)"
        },
        title="Model Accuracy Comparison"
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("---")

st.header("🤖 Model Performance")


performance_rows = []

for name, values in metrics["results"].items():

    performance_rows.append({

        "Model": name,

        "Accuracy":
            f"{values['accuracy'] * 100:.2f}%",

        "Precision":
            f"{values['precision'] * 100:.2f}%",

        "Recall":
            f"{values['recall'] * 100:.2f}%",

        "F1 Score":
            f"{values['f1_score'] * 100:.2f}%",

        "ROC-AUC":
            f"{values['roc_auc']:.4f}"
    })


performance_df = pd.DataFrame(
    performance_rows
)


st.dataframe(
    performance_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PREDICTION SECTION
# ============================================================

st.markdown("---")

st.header("🔮 Predict Customer Churn")

st.write(
    "Enter customer information to estimate "
    "the probability of churn."
)


left, right = st.columns(2)


# ============================================================
# LEFT INPUTS
# ============================================================

with left:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        24
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "Yes",
            "No",
            "No phone service"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


# ============================================================
# RIGHT INPUTS
# ============================================================

with right:

    online_backup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    device_protection = st.selectbox(
        "Device Protection",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


charge1, charge2 = st.columns(2)


with charge1:

    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )


with charge2:

    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=2000.0
    )


predict_button = st.button(
    "🚀 Predict Churn",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    input_df = create_input(

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
    )

    prediction, probability = predict(
        input_df
    )

    risk = risk_level(
        probability
    )

    st.markdown("---")

    st.header("🎯 Prediction Result")


    result1, result2 = st.columns(2)


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    with result1:

        if prediction == 1:

            st.error(
                "⚠ High Probability of Churn"
            )

        else:

            st.success(
                "✅ Customer Likely to Stay"
            )


        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )


        if risk == "LOW":

            st.success(
                f"Risk Level: {risk}"
            )

        elif risk == "MEDIUM":

            st.warning(
                f"Risk Level: {risk}"
            )

        else:

            st.error(
                f"Risk Level: {risk}"
            )


        st.progress(
            min(
                max(probability, 0.0),
                1.0
            )
        )


    # --------------------------------------------------------
    # GAUGE
    # --------------------------------------------------------

    with result2:

        gauge = go.Figure(
            go.Indicator(

                mode="gauge+number",

                value=probability * 100,

                title={
                    "text":
                    "Churn Probability (%)"
                },

                gauge={

                    "axis": {
                        "range": [0, 100]
                    },

                    "steps": [

                        {
                            "range": [0, 30],
                            "color": "#16a34a"
                        },

                        {
                            "range": [30, 70],
                            "color": "#eab308"
                        },

                        {
                            "range": [70, 100],
                            "color": "#dc2626"
                        }

                    ]

                }
            )
        )

        gauge.update_layout(
            height=350
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )


    # ========================================================
    # CUSTOMER SUMMARY
    # ========================================================

    st.markdown("---")

    st.subheader("👤 Customer Summary")


    s1, s2, s3 = st.columns(3)


    with s1:

        st.info(
            f"Gender: {gender}"
        )

        st.info(
            f"Partner: {partner}"
        )

        st.info(
            f"Dependents: {dependents}"
        )


    with s2:

        st.info(
            f"Tenure: {tenure} months"
        )

        st.info(
            f"Internet: {internet_service}"
        )

        st.info(
            f"Contract: {contract}"
        )


    with s3:

        st.info(
            f"Monthly Charges: ${monthly:.2f}"
        )

        st.info(
            f"Total Charges: ${total:.2f}"
        )

        st.info(
            f"Payment: {payment}"
        )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.markdown("---")

    st.subheader(
        "💡 Business Recommendations"
    )


    recommendations = recommendation(
        probability
    )


    for item in recommendations:

        st.success(
            "✓ " + item
        )


# ============================================================
# ABOUT
# ============================================================

st.markdown("---")

st.header("📘 About This Project")


about1, about2 = st.columns(2)


with about1:

    st.write(
        """
        **Customer Churn Prediction** is a machine
        learning application designed to identify
        telecom customers who may discontinue their
        services.

        The project uses customer demographics,
        account information, service subscriptions
        and billing information to estimate churn
        probability.

        The system compares multiple classification
        algorithms and provides business-oriented
        retention recommendations.
        """
    )


with about2:

    st.info(
        """
        **Technology Stack**

        🐍 Python

        📊 Pandas

        🤖 Scikit-learn

        🌲 Random Forest

        ⚡ XGBoost

        📈 Plotly

        🌐 Streamlit

        💾 Joblib
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        padding:25px;
        border-radius:15px;
        background:rgba(255,255,255,0.06);
    ">

    <h3>📊 Customer Churn Prediction</h3>

    <p>
    Machine Learning & Data Analytics Project
    </p>

    <p>
    Developed by <b>Khemraj Bhagat</b>
    </p>

    </div>
    """,
    unsafe_allow_html=True
)