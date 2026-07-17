# ============================================================
# Customer Churn Prediction Dashboard
# Part 1
# ============================================================

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from utils import create_input, predict, risk_level, recommendation

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Load CSS
# ------------------------------------------------------------

def load_css():
    with open("style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.markdown("""
<div class="main-title">
📊 Customer Churn Prediction
</div>
<div class="sub-title">
AI Powered Customer Retention System
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------------------
# KPI Cards
# ------------------------------------------------------------

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        label="👥 Customers",
        value="7043"
    )

with k2:
    st.metric(
        label="🎯 Accuracy",
        value="80.31%"
    )

with k3:
    st.metric(
        label="⚠ Churn Rate",
        value="26.54%"
    )

with k4:
    st.metric(
        label="🤖 Best Model",
        value="Logistic Regression"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Dashboard Charts
# ------------------------------------------------------------

left_chart, right_chart = st.columns(2)

with left_chart:

    st.subheader("🥧 Customer Distribution")

    pie = go.Figure(
        data=[
            go.Pie(
                labels=["Retained","Churn"],
                values=[5174,1869],
                hole=0.55,
                marker=dict(
                    colors=[
                        "#00E676",
                        "#FF5252"
                    ]
                )
            )
        ]
    )

    pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        height=420
    )

    st.plotly_chart(
        pie,
        use_container_width=True
    )

with right_chart:

    st.subheader("📈 Model Comparison")

    models = [
        "Logistic Regression",
        "Random Forest"
    ]

    accuracy = [
        80.31,
        78.96
    ]

    bar = go.Figure()

    bar.add_trace(

        go.Bar(

            x=models,

            y=accuracy,

            text=accuracy,

            textposition="outside",

            marker_color=[
                "#00E5FF",
                "#7C4DFF"
            ]

        )

    )

    bar.update_layout(

        yaxis_title="Accuracy",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font_color="white",

        height=420

    )

    st.plotly_chart(
        bar,
        use_container_width=True
    )

st.markdown("---")

st.header("👤 Customer Information")

left, right = st.columns(2)
# ============================================================
# Customer Input Form
# ============================================================

with left:

    gender = st.selectbox(
        "👤 Gender",
        ["Male", "Female"]
    )

    senior = st.selectbox(
        "🧓 Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "❤️ Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "👨‍👩‍👧 Dependents",
        ["Yes", "No"]
    )

    tenure = st.slider(
        "📅 Tenure (Months)",
        0,
        72,
        24
    )

    phone_service = st.selectbox(
        "☎ Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "📞 Multiple Lines",
        [
            "Yes",
            "No",
            "No phone service"
        ]
    )

    internet_service = st.selectbox(
        "🌐 Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    online_security = st.selectbox(
        "🔒 Online Security",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

with right:

    online_backup = st.selectbox(
        "💾 Online Backup",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    device_protection = st.selectbox(
        "🛡 Device Protection",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "🛠 Tech Support",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "📺 Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "🎬 Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    contract = st.selectbox(
        "📝 Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless = st.selectbox(
        "📄 Paperless Billing",
        [
            "Yes",
            "No"
        ]
    )

    payment = st.selectbox(
        "💳 Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )

st.markdown("---")

c1, c2 = st.columns(2)

with c1:

    monthly = st.slider(
        "💰 Monthly Charges",
        0.0,
        150.0,
        70.0
    )

with c2:

    total = st.slider(
        "💵 Total Charges",
        0.0,
        9000.0,
        2000.0
    )

st.markdown("<br>", unsafe_allow_html=True)

predict_btn = st.button(
    "🚀 Predict Customer Churn"
)

st.markdown("---")
# ============================================================
# Prediction
# ============================================================

if predict_btn:

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

    prediction, probability = predict(input_df)

    st.header("📊 Prediction Result")

    result1, result2 = st.columns([1, 1.2])

    # --------------------------------------------------------
    # Left Side
    # --------------------------------------------------------

    with result1:

        if prediction == 1:

            st.error("⚠ Customer is likely to Churn")

        else:

            st.success("✅ Customer is NOT likely to Churn")

        st.metric(
            "Churn Probability",
            f"{probability*100:.2f}%"
        )

        st.metric(
            "Risk Level",
            risk_level(probability)
        )

        st.progress(float(probability))

    # --------------------------------------------------------
    # Right Side
    # --------------------------------------------------------

    with result2:

        gauge = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=probability*100,

                title={
                    "text":"Prediction Confidence"
                },

                gauge={

                    "axis":{
                        "range":[0,100]
                    },

                    "bar":{
                        "color":"deepskyblue"
                    },

                    "steps":[

                        {
                            "range":[0,30],
                            "color":"green"
                        },

                        {
                            "range":[30,70],
                            "color":"gold"
                        },

                        {
                            "range":[70,100],
                            "color":"red"
                        }

                    ]

                }

            )

        )

        gauge.update_layout(

            paper_bgcolor="rgba(0,0,0,0)",

            font_color="white",

            height=350

        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Customer Summary
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("👤 Customer Summary")

    s1, s2, s3 = st.columns(3)

    with s1:

        st.info(f"Gender : {gender}")
        st.info(f"Partner : {partner}")
        st.info(f"Dependents : {dependents}")

    with s2:

        st.info(f"Tenure : {tenure} Months")
        st.info(f"Internet : {internet_service}")
        st.info(f"Contract : {contract}")

    with s3:

        st.info(f"Monthly Charges : ${monthly}")
        st.info(f"Total Charges : ${total}")
        st.info(f"Payment : {payment}")

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("💡 AI Recommendation")

    rec = recommendation(probability)

    for item in rec:

        st.success(item)

    st.markdown("---")

    st.subheader("📈 Business Insight")

    if probability < 0.30:

        st.success("""
        Customer has a very low probability of churn.

        Recommended Action:

        • Maintain regular communication

        • Offer loyalty rewards

        • Upsell premium plans
        """)

    elif probability < 0.70:

        st.warning("""
        Customer has a medium risk of churn.

        Recommended Action:

        • Send promotional discounts

        • Improve customer engagement

        • Personalized offers
        """)

    else:

        st.error("""
        Customer has a HIGH probability of churn.

        Recommended Action:

        • Contact immediately

        • Assign retention specialist

        • Offer contract upgrade

        • Provide special discounts
        """)
        # ============================================================
# About Project
# ============================================================

st.markdown("---")

st.header("📘 About This Project")

left, right = st.columns([2,1])

with left:

    st.write("""
### 📊 Customer Churn Prediction using Machine Learning

This application predicts whether a telecom customer is likely to leave a company based on customer demographics, account information, and subscribed services.

The objective is to help businesses identify high-risk customers and improve customer retention strategies.

---

### 🚀 Technologies Used

- 🐍 Python
- 📊 Pandas
- 🤖 Scikit-Learn
- 📈 Plotly
- 🌐 Streamlit
- 💾 Joblib

---

### 🤖 Machine Learning Models

✔ Logistic Regression

✔ Random Forest

---

### 📂 Dataset

**Telco Customer Churn Dataset**

- Total Customers : **7043**
- Features : **20+**
- Target Variable : **Churn**

---

### 📏 Model Evaluation

✔ Accuracy

✔ Precision

✔ Recall

✔ F1 Score

✔ ROC-AUC

---

### 🎯 Project Goal

To build an intelligent decision support system that predicts customer churn and helps companies improve customer retention.

""")

with right:

    st.metric("Dataset Size", "7043")

    st.metric("Features", "30")

    st.metric("Algorithms", "2")

    st.metric("Accuracy", "80.31%")

    st.metric("Best Model", "Logistic Regression")

# ============================================================
# Feature Importance (Illustration)
# ============================================================

st.markdown("---")

st.subheader("📊 Important Features")

feature_data = {
    "Feature":[
        "Monthly Charges",
        "Tenure",
        "Contract",
        "Internet Service",
        "Payment Method",
        "Tech Support",
        "Online Security"
    ],

    "Importance":[
        0.95,
        0.88,
        0.80,
        0.71,
        0.63,
        0.58,
        0.52
    ]
}

feature_df = pd.DataFrame(feature_data)

fig = px.bar(

    feature_df,

    x="Importance",

    y="Feature",

    orientation="h",

    color="Importance",

    color_continuous_scale="Blues"

)

fig.update_layout(

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)",

    font_color="white",

    height=450,

    yaxis_title="",

    xaxis_title="Importance Score"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# Footer
# ============================================================

st.markdown("---")

st.markdown("""

<div style='text-align:center;
padding:25px;
background:rgba(255,255,255,0.08);
border-radius:15px;'>

<h2>🚀 Customer Churn Prediction Dashboard</h2>

<p style='font-size:18px;'>

Developed using ❤️ with

<b>Python • Streamlit • Scikit-Learn • Plotly</b>

</p>

<p>

Machine Learning Internship Project

</p>

<p>

© 2026 All Rights Reserved

</p>

</div>

""",unsafe_allow_html=True)