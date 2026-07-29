import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("diabetes_model (1).pkl")
scaler = joblib.load("scaler (1).pkl")

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_excel("diabetes (1).csv.xlsx")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("🩺 Diabetes Prediction System")

st.sidebar.markdown("## Patient Information")

preg = st.sidebar.number_input(
    "Pregnancies",
    min_value=0,
    value=2
)

glucose = st.sidebar.number_input(
    "Glucose (mg/dL)",
    min_value=0,
    value=120
)

bp = st.sidebar.number_input(
    "Blood Pressure (mm Hg)",
    min_value=0,
    value=70
)

skin = st.sidebar.number_input(
    "Skin Thickness (mm)",
    min_value=0,
    value=20
)

insulin = st.sidebar.number_input(
    "Insulin (mu U/ml)",
    min_value=0,
    value=85
)

bmi = st.sidebar.number_input(
    "BMI (kg/m²)",
    min_value=0.0,
    value=28.5
)

dpf = st.sidebar.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.35
)

age = st.sidebar.number_input(
    "Age (years)",
    min_value=1,
    value=45
)

predict = st.sidebar.button("🔍 Predict Diabetes Risk")

# -------------------------------
# MAIN TITLE
# -------------------------------
st.title("🩺 Diabetes Prediction Dashboard")

st.subheader("Using Machine Learning")

st.markdown("---")
# ============================
# PREDICTION
# ============================

if predict:

    patient = np.array([[preg, glucose, bp, skin,
                         insulin, bmi, dpf, age]])

    patient = scaler.transform(patient)

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1] * 100

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🔴 High Risk of Diabetes")
        else:
            st.success("🟢 Non-Diabetic")

        st.metric(
            "Probability of Diabetes",
            f"{probability:.2f}%"
        )

    with col2:

        st.subheader("Risk Level")

        st.progress(int(probability))

        st.write(f"Risk Score : {probability:.2f}%")
        )

    with col2:

        st.subheader("Risk Level")

        st.progress(int(probability))

       st.write(f"Risk Score : {probability:.2f}%")

st.subheader("🩺 Health Recommendation")

if probability >= 70:
    st.error("High Risk")
    st.write("• Consult a doctor as soon as possible.")
    st.write("• Monitor your blood glucose regularly.")
    st.write("• Follow a diabetic-friendly diet.")
    st.write("• Exercise for at least 30 minutes daily.")
    st.write("• Reduce sugar and processed foods.")

elif probability >= 40:
    st.warning("Moderate Risk")
    st.write("• Maintain a balanced diet.")
    st.write("• Exercise regularly.")
    st.write("• Check your blood sugar periodically.")
    st.write("• Maintain a healthy body weight.")

else:
    st.success("Low Risk")
    st.write("• Continue your healthy lifestyle.")
    st.write("• Eat a balanced diet.")
    st.write("• Stay physically active.")
    st.write("• Have regular health check-ups.")
