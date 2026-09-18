import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

st.title("Customer Churn Prediction with Explainable AI (XAI)")
st.write("Welcome! Fill in the customer details below to predict churn.")

# ---------------------------------------------------------------------------
# SIDEBAR — PROJECT INFO
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ About This App")
    st.write(
        "This app predicts whether a telecom customer is likely to churn, "
        "and explains why using SHAP (Explainable AI)."
    )

    st.subheader("Model Details")
    st.write("- Algorithm: Random Forest Classifier")
    st.write("- Explainability: SHAP")
    st.write("- Dataset: IBM Telco Customer Churn")

    st.subheader("How to Use")
    st.write(
        "1. Fill in the customer details.\n"
        "2. Click Predict.\n"
        "3. View the churn prediction and probability.\n"
        "4. Expand 'Why this prediction?' for the SHAP explanation."
    )

# ---------------------------------------------------------------------------
# LOAD MODEL & ENCODERS
# ---------------------------------------------------------------------------
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

# ---------------------------------------------------------------------------
# MODEL PERFORMANCE — accuracy, precision, recall, F1, confusion matrix
# ---------------------------------------------------------------------------
# Recomputed on the same 80:20 split (random_state=42) used in train_model.py,
# so these numbers match the training script's console output exactly.
DATA_PATH = "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"


@st.cache_data
def get_test_performance():
    df = pd.read_csv(DATA_PATH)
    df = df.drop("customerID", axis=1)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "cm": confusion_matrix(y_test, y_pred),
    }
    return metrics


with st.expander("Model Performance & Confusion Matrix (Test Set)"):
    try:
        perf = get_test_performance()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{perf['accuracy']:.2%}")
        col2.metric("Precision", f"{perf['precision']:.2%}")
        col3.metric("Recall", f"{perf['recall']:.2%}")
        col4.metric("F1-score", f"{perf['f1']:.2%}")

        st.write("**Confusion Matrix**")
        cm = perf["cm"]
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Churn", "Churn"])
        ax.set_yticklabels(["No Churn", "Churn"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
        st.pyplot(fig)

    except FileNotFoundError:
        st.info(f"Dataset not found at '{DATA_PATH}'. Update DATA_PATH to see performance metrics.")

# ---------------------------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------------------------
st.subheader("Personal Information")
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
with col2:
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)

st.subheader("Services & Subscription Details")
col1, col2 = st.columns(2)
with col1:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
with col2:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.subheader("Billing & Contract Information")
col1, col2 = st.columns(2)
with col1:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
with col2:
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

# ---------------------------------------------------------------------------
# PREDICT
# ---------------------------------------------------------------------------
if st.button("Predict"):

    input_dict = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
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
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([input_dict])

    for col, le in encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col].astype(str))

    with st.spinner("Predicting..."):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"This customer is likely to CHURN. (Probability: {probability:.2%})")
    else:
        st.success(f"This customer is likely to STAY. (Churn probability: {probability:.2%})")

    st.metric("Churn Probability", f"{probability:.1%}")
    st.progress(float(probability))

    with st.expander("Why this prediction? (SHAP for this customer)"):
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        expected_value = explainer.expected_value

        if isinstance(shap_values, list):
            values_for_churn = shap_values[1][0]
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            values_for_churn = shap_values[0, :, 1]
        else:
            values_for_churn = shap_values[0]

        if hasattr(expected_value, "__len__"):
            base_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        else:
            base_value = expected_value

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots._waterfall.waterfall_legacy(
            base_value,
            values_for_churn,
            feature_names=input_df.columns.tolist(),
            show=False
        )
        st.pyplot(fig)
        st.caption("Red = pushes toward churn, Blue = pushes toward staying")

    with st.expander("Overall feature importance (from training data)"):
        st.image("shap_summary.png", caption="SHAP Summary Plot (training data)")
