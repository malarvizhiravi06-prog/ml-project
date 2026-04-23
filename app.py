import streamlit as st
import pandas as pd
import pickle
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
    columns = pickle.load(open('columns.pkl', 'rb'))
    st.success("Models loaded successfully ✅")
except:
    st.error("Error loading model files")
    st.stop()

# -------------------------
# INPUT SECTION
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
st.header("🧠 Financial Literacy Assessment")

questions = [
    "Do you assess repayment ability before taking loan?",
    "Do you track expenses regularly?",
    "Do you understand interest rates?",
    "Do you evaluate financial risks?",
    "Do you plan finances for future?",
    "Do you understand investments?",
    "Do you repay dues on time?"
]

answers = []
for q in questions:
    ans = st.radio(q, ["Yes", "No"], horizontal=True)
    answers.append(ans)

bad_decision = st.radio(
    "Do you make financial decisions without proper analysis?",
    ["Yes", "No"], horizontal=True
)

# -------------------------
# PREDICTION
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
    # MODEL INPUT
    # -------------------------
    data = {
        'Gender': 1 if Gender == "Male" else 0,
        'Married': 1 if Married == "Yes" else 0,
        'Dependents': int(Dependents.replace("+", "")),
        'Education': 1 if Education == "Graduate" else 0,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': 1
    }

    df = pd.DataFrame([data])

    # Feature Engineering
    df['Income_Total'] = ApplicantIncome + CoapplicantIncome
    df['Loan_Income_Ratio'] = LoanAmount / (df['Income_Total'] + 1)
    df['Income_Percentile'] = 2
    df['Loan_Percentile'] = 2
    df['Cluster'] = 1

# -------------------------
# DATA ALIGNMENT
# -------------------------
df = df.reindex(columns=columns)
df = df.fillna(0)

# -------------------------
# SCALING
# -------------------------
df_scaled = None

try:
    df_scaled = scaler.transform(df)
except Exception as e:
    st.error(f"Scaling failed: {e}")
    st.stop()

# -------------------------
# SAFETY CHECK
# -------------------------
if df_scaled is None:
    st.error("Scaling failed. Cannot proceed.")
    st.stop()

# -------------------------
# MODEL OUTPUT
# -------------------------
pred = model_class.predict(df_scaled)[0]
prob = model_class.predict_proba(df_scaled)[0][1]

st.header("📊 Results")

if pred == 1:
        st.success("Loan Approved ✅")
        amount = model_reg.predict(df_scaled)[0]
        st.metric("Predicted Loan Amount", f"{amount:.2f}")
else:
        st.error("Loan Rejected ❌")

st.write(f"Approval Probability: {prob*100:.2f}%")

    # -------------------------
    # LITERACY OUTPUT
    # -------------------------
st.subheader("🧠 Financial Literacy")

st.write(f"Score: {literacy_score:.2f}/100")
st.write(f"Level: {level}")

    # -------------------------
    # RISK
    # -------------------------
st.subheader("⚠️ Risk Interpretation")

if level == "Low":
        st.error("High behavioural risk due to low financial awareness.")
elif level == "Medium":
        st.warning("Moderate financial awareness with some risks.")
else:
        st.success("Low behavioural risk with good financial understanding.")

    # -------------------------
    # RECOMMENDATIONS
    # -------------------------
st.subheader("💡 Recommendations")

if level == "Low":
        st.write("- Improve budgeting and expense tracking")
        st.write("- Understand loan repayment obligations")
elif level == "Medium":
        st.write("- Improve financial planning and risk assessment")
else:
        st.write("- Maintain strong financial discipline")

    # -------------------------
    # REASONS
    # -------------------------
if pred == 0:
        st.subheader("❗ Possible Reasons")

        reasons = []

        if Credit_History == 0:
            reasons.append("Poor credit history")
        if ApplicantIncome < 3000:
            reasons.append("Low income")
        if LoanAmount > ApplicantIncome:
            reasons.append("High loan burden")

        for r in reasons:
            st.write("-", r)

    # -------------------------
    # CHART
    # -------------------------
st.subheader("📊 Literacy Score Visualization")

fig, ax = plt.subplots()
ax.bar(["Score"], [literacy_score])
ax.set_ylim(0, 100)
st.pyplot(fig)
