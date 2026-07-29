import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model and scaler
model = joblib.load("diabetes_model (1).pkl")
scaler = joblib.load("scaler (1).pkl")

df = pd.read_excel("diabetes (1).csv.xlsx")

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("🩺 Intelligent Diabetes Prediction System")
st.write("Enter the patient's details below.")

preg = st.number_input("Pregnancies", min_value=0, value=1)
glucose = st.number_input("Glucose", min_value=0, value=120)
bp = st.number_input("Blood Pressure", min_value=0, value=70)
skin = st.number_input("Skin Thickness", min_value=0, value=20)
insulin = st.number_input("Insulin", min_value=0, value=80)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
age = st.number_input("Age", min_value=1, value=30)

if st.button("Predict"):

    patient = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

    patient = scaler.transform(patient)

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1] * 100

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Diabetic")
    else:
        st.success("✅ Non-Diabetic")

    st.write(f"Risk Probability: {probability:.2f}%")

    st.subheader("Health Recommendation")

    if probability >= 70:
        st.warning("High Risk")
        st.write("• Consult a doctor immediately.")
        st.write("• Exercise regularly.")
        st.write("• Reduce sugar intake.")
        st.write("• Monitor blood glucose frequently.")

    elif probability >= 40:
        st.info("Moderate Risk")
        st.write("• Maintain a healthy diet.")
        st.write("• Walk at least 30 minutes daily.")
        st.write("• Monitor your blood sugar regularly.")

    else:
        st.success("Low Risk")
        st.write("• Continue a healthy lifestyle.")
        st.write("• Exercise regularly.")
        st.write("• Eat a balanced diet.")
      
import plotly.express as px

st.header("📊 Interactive Dashboard")

fig = px.histogram(df, x="Glucose", title="Glucose Distribution")
st.plotly_chart(fig)
