
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Loan Risk Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# LOAD MODELS
# =========================================

rf_model = joblib.load("random_forest.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
kmeans = joblib.load("kmeans.pkl")

# =========================================
# CUSTOM CSS
# =========================================

st.markdown(
    """
    <style>

    .main {
        background: linear-gradient(to bottom right, #0f172a, #111827);
        color: white;
    }

    .stApp {
        background: linear-gradient(to bottom right, #0f172a, #111827);
    }

    .hero-card {
        background: linear-gradient(135deg, #1e293b, #111827);
        padding: 35px;
        border-radius: 22px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0px 10px 30px rgba(0,0,0,0.4);
        margin-bottom: 25px;
    }

    .metric-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
    }

    .prediction-approved {
        background: linear-gradient(135deg, #16a34a, #15803d);
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        color: white;
        font-size: 28px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
    }

    .prediction-rejected {
        background: linear-gradient(135deg, #dc2626, #991b1b);
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        color: white;
        font-size: 28px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
    }

    .section-title {
        font-size: 28px;
        font-weight: 700;
        color: white;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    div[data-baseweb="select"] > div {
        background-color: #1e293b;
        color: white;
        border-radius: 10px;
    }

    .stSlider > div > div {
        color: #38bdf8;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================
# SIDEBAR
# =========================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
        width=120
    )

    st.markdown("## Loan Risk Intelligence")

    st.markdown("---")

    st.write("### System Features")

    st.success("✔ Loan Approval Prediction")
    st.success("✔ Credit Risk Analysis")
    st.success("✔ Customer Segmentation")
    st.success("✔ AI-Powered Decision Making")
    st.success("✔ PCA Feature Optimization")

    st.markdown("---")

    st.write("### Developed For")
    st.info("Banking & Financial Institutions")

    st.markdown("---")

    st.caption(f"Date: {datetime.now().strftime('%d %B %Y')}")

# =========================================
# HERO SECTION
# =========================================

st.markdown(
    """
    <div class='hero-card'>
        <h1 style='font-size:48px;'>💳 AI Loan Approval Prediction System</h1>
        <p style='font-size:20px;color:#cbd5e1;'>
            Advanced machine learning platform for loan approval prediction,
            credit risk assessment, and intelligent customer segmentation.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================
# METRICS SECTION
# =========================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class='metric-card'>
            <h2>98%</h2>
            <p>Prediction Accuracy</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class='metric-card'>
            <h2>AI</h2>
            <p>Risk Detection</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class='metric-card'>
            <h2>24/7</h2>
            <p>Smart Automation</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        """
        <div class='metric-card'>
            <h2>Secure</h2>
            <p>Enterprise Ready</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================
# FORM SECTION
# =========================================

st.markdown("<div class='section-title'>Customer Information</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age", 18, 70, 30)

    income = st.number_input(
        "Annual Income",
        min_value=1000,
        max_value=1000000,
        value=50000,
        step=1000
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=1000,
        max_value=500000,
        value=10000,
        step=1000
    )

    credit_score = st.slider(
        "Credit Score",
        300,
        900,
        700
    )

    months_employed = st.slider(
        "Months Employed",
        0,
        500,
        60
    )

    num_credit_lines = st.slider(
        "Number of Credit Lines",
        1,
        20,
        5
    )

with col2:

    interest_rate = st.slider(
        "Interest Rate",
        1.0,
        25.0,
        10.0
    )

    loan_term = st.slider(
        "Loan Term (Months)",
        6,
        360,
        60
    )

    dti_ratio = st.slider(
        "Debt-to-Income Ratio",
        0.0,
        1.0,
        0.3
    )

    education = st.selectbox(
        "Education",
        ["High School", "Bachelor", "Master", "PhD"]
    )

    employment_type = st.selectbox(
        "Employment Type",
        ["Full-time", "Part-time", "Self-employed", "Unemployed"]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

# =========================================
# ADDITIONAL INFO
# =========================================

col3, col4, col5, col6 = st.columns(4)

with col3:
    has_mortgage = st.selectbox("Has Mortgage", ["Yes", "No"])

with col4:
    has_dependents = st.selectbox("Has Dependents", ["Yes", "No"])

with col5:
    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Home", "Business", "Education", "Auto", "Personal"]
    )

with col6:
    has_cosigner = st.selectbox("Has Co-Signer", ["Yes", "No"])

# =========================================
# ENCODING MAPS
# =========================================

education_map = {
    "Bachelor": 0,
    "High School": 1,
    "Master": 2,
    "PhD": 3
}

employment_map = {
    "Full-time": 0,
    "Part-time": 1,
    "Self-employed": 2,
    "Unemployed": 3
}

marital_map = {
    "Divorced": 0,
    "Married": 1,
    "Single": 2
}

binary_map = {
    "No": 0,
    "Yes": 1
}

loan_purpose_map = {
    "Auto": 0,
    "Business": 1,
    "Education": 2,
    "Home": 3,
    "Personal": 4
}

# =========================================
# PREDICT BUTTON
# =========================================

if st.button("🚀 Predict Loan Risk", use_container_width=True):

    input_data = pd.DataFrame([[
        age,
        income,
        loan_amount,
        credit_score,
        months_employed,
        num_credit_lines,
        interest_rate,
        loan_term,
        dti_ratio,
        education_map[education],
        employment_map[employment_type],
        marital_map[marital_status],
        binary_map[has_mortgage],
        binary_map[has_dependents],
        loan_purpose_map[loan_purpose],
        binary_map[has_cosigner]
    ]], columns=[
        'Age',
        'Income',
        'LoanAmount',
        'CreditScore',
        'MonthsEmployed',
        'NumCreditLines',
        'InterestRate',
        'LoanTerm',
        'DTIRatio',
        'Education',
        'EmploymentType',
        'MaritalStatus',
        'HasMortgage',
        'HasDependents',
        'LoanPurpose',
        'HasCoSigner'
    ])

    # PREPROCESS

    scaled = scaler.transform(input_data)
    transformed = pca.transform(scaled)

    # PREDICTIONS

    prediction = rf_model.predict(transformed)[0]

    probability = rf_model.predict_proba(transformed)[0][1]

    cluster = kmeans.predict(transformed)[0]

    # RISK LEVEL

    if probability < 0.30:
        risk = "LOW RISK"
        risk_color = "green"

    elif probability < 0.70:
        risk = "MEDIUM RISK"
        risk_color = "orange"

    else:
        risk = "HIGH RISK"
        risk_color = "red"

    # RESULT UI

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction == 0:

        st.markdown(
            f"""
            <div class='prediction-approved'>
                ✅ LOAN APPROVED
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class='prediction-rejected'>
                ❌ LOAN REJECTED
            </div>
            """,
            unsafe_allow_html=True
        )

    # RESULT CARDS

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Risk Probability",
            value=f"{round(probability*100,2)}%"
        )

    with col2:
        st.metric(
            label="Customer Segment",
            value=f"Segment {cluster}"
        )

    with col3:
        st.metric(
            label="Risk Category",
            value=risk
        )

    # GAUGE CHART

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        title = {'text': "Loan Risk Score"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': 'green'},
                {'range': [30, 70], 'color': 'orange'},
                {'range': [70, 100], 'color': 'red'}
            ]
        }
    ))

    fig.update_layout(
        height=400,
        paper_bgcolor='#111827',
        font={'color': 'white'}
    )

    st.plotly_chart(fig, use_container_width=True)

    # CUSTOMER INSIGHTS

    st.markdown("## 📊 AI Insights")

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:

        st.info(
            f"""
            Credit Score Analysis:
            Current credit score is {credit_score}.
            Higher credit scores improve loan approval chances.
            """
        )

    with insight_col2:

        st.warning(
            f"""
            Debt-to-Income Ratio:
            Current DTI Ratio is {dti_ratio}.
            Lower DTI indicates better repayment capability.
            """
        )

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.markdown(
    """
    <center>
        <p style='color:gray;'>
        AI Loan Risk Prediction System | Built with Streamlit, Machine Learning & Advanced Analytics
        </p>
    </center>
    """,
    unsafe_allow_html=True
)
