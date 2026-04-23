import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Financial Decision System", layout="wide")

st.title("📊 Financial Decision Support System")

# -------------------------
# LOAD MODELS
# -------------------------
try:
    model_class = pickle.load(open('model_class.pkl', 'rb'))
    model_reg = pickle.load(open('model_reg.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    st.success("Models Loaded Successfully ✅")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# -------------------------
# INPUTS
# -------------------------
st.header("👤 Applicant Details")

col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

with col2:
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.header("💰 Financial Details")

col3, col4 = st.columns(2)

with col3:
    ApplicantIncome = st.number_input("Applicant Income", 0)
    LoanAmount = st.number_input("Loan Amount", 0)

with col4:
    CoapplicantIncome = st.number_input("Coapplicant Income", 0)
    Loan_Amount_Term = st.number_input("Loan Term", 0)

credit = st.selectbox("Credit History", ["Good", "Bad"])
Credit_History = 1 if credit == "Good" else 0

# -------------------------
# LITERACY QUESTIONS
# -------------------------
st.header("🧠 Financial Literacy")

questions = [
    "Do you assess repayment ability before loan?",
    "Do you track expenses?",
    "Do you understand interest rates?",
    "Do you evaluate risks?",
    "Do you plan finances?",
    "Do you understand investments?",
    "Do you repay dues on time?"
]

answers = []
for q in questions:
    ans = st.radio(q, ["Yes", "No"], horizontal=True)
    answers.append(ans)

bad_decision = st.radio(
    "Do you make financial decisions without analysis?",
    ["Yes", "No"], horizontal=True
)

# -------------------------
# PREDICT
# -------------------------
if st.button("🔍 Predict"):

    # -------------------------
    # LITERACY SCORE
    # -------------------------
    score = sum([1 for a in answers if a == "Yes"])

    if bad_decision == "Yes":
        score -= 1

    if Education == "Graduate":
        score += 1

    literacy_score = max(0, min(100, (score / 8) * 100))

    if literacy_score < 40:
        level = "Low"
    elif literacy_score < 70:
        level = "Medium"
    else:
        level = "High"

    # -------------------------
    # INPUT DATA (BASE 8)
    # -------------------------
    input_data = [
        1 if Gender == "Male" else 0,
        1 if Married == "Yes" else 0,
        int(Dependents.replace("+", "")),
        1 if Education == "Graduate" else 0,
        ApplicantIncome,
        CoapplicantIncome,
        Loan_Amount_Term,
        Credit_History
    ]

    # -------------------------
    # MATCH SCALER (16 FEATURES)
    # -------------------------
    arr = np.array(input_data)

    if len(arr) < scaler.n_features_in_:
        arr = np.pad(arr, (0, scaler.n_features_in_ - len(arr)))
    else:
        arr = arr[:scaler.n_features_in_]

    df = arr.reshape(1, -1)

    # -------------------------
    # SCALE (16 FEATURES)
    # -------------------------
    try:
        df_scaled = scaler.transform(df)
    except Exception as e:
        st.error(f"Scaling failed: {e}")
        st.stop()

    # -------------------------
    # MATCH MODEL (17 FEATURES)
    # -------------------------
    if df_scaled.shape[1] < model_class.n_features_in_:
        df_scaled = np.pad(
            df_scaled,
            ((0, 0), (0, model_class.n_features_in_ - df_scaled.shape[1])),
            mode='constant'
        )

    # -------------------------
    # PREDICTION
    # -------------------------
    try:
        pred = model_class.predict(df_scaled)[0]
        prob = model_class.predict_proba(df_scaled)[0][1]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.header("📊 Results")

    if pred == 1:
        st.success("Loan Approved ✅")
        try:
            amount = model_reg.predict(df_scaled)[0]
            st.metric("Estimated Loan Amount", f"{amount:.2f}")
        except:
            pass
    else:
        st.error("Loan Rejected ❌")

    st.write(f"Approval Probability: {prob*100:.2f}%")

    # -------------------------
    # LITERACY OUTPUT
    # -------------------------
    st.subheader("🧠 Literacy Score")
    st.write(f"{literacy_score:.2f} / 100 ({level})")

    # -------------------------
    # RISK
    # -------------------------
    st.subheader("⚠️ Risk Interpretation")

    if level == "Low":
        st.error("High financial risk")
    elif level == "Medium":
        st.warning("Moderate risk")
    else:
        st.success("Low risk")

    # -------------------------
    # CHART
    # -------------------------
    st.subheader("📊 Literacy Score Chart")

    fig, ax = plt.subplots()
    ax.bar(["Score"], [literacy_score])
    ax.set_ylim(0, 100)
    st.pyplot(fig)
