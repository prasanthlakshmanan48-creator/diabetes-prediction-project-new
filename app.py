import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================

model = joblib.load("diabetes_model (1).pkl")
scaler = joblib.load("scaler (1).pkl")

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_excel("diabetes (1).csv.xlsx")

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("🩺 Diabetes Prediction System")
st.sidebar.markdown("### Enter Patient Details")

preg = st.sidebar.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=2
)

glucose = st.sidebar.number_input(
    "Glucose (mg/dL)",
    min_value=0,
    max_value=250,
    value=120
)

bp = st.sidebar.number_input(
    "Blood Pressure (mmHg)",
    min_value=0,
    max_value=150,
    value=70
)

skin = st.sidebar.number_input(
    "Skin Thickness (mm)",
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.sidebar.number_input(
    "Insulin (μU/mL)",
    min_value=0,
    max_value=900,
    value=85
)

bmi = st.sidebar.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=28.5
)

dpf = st.sidebar.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.35
)

age = st.sidebar.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=45
)

predict = st.sidebar.button("🔍 Predict Diabetes Risk")

# =====================================
# MAIN TITLE
# =====================================

st.title("🩺 Intelligent Diabetes Prediction System")
st.write("Machine Learning Based Diabetes Risk Prediction Dashboard")

st.markdown("---")

# =====================================
# KPI CARDS
# =====================================

total_records = len(df)
diabetic = len(df[df["Outcome"] == 1])
non_diabetic = len(df[df["Outcome"] == 0])

accuracy = 83.2

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("📋 Total Patients", total_records)

with c2:
    st.metric("🟢 Non-Diabetic", non_diabetic)

with c3:
    st.metric("🔴 Diabetic", diabetic)

with c4:
    st.metric("🎯 Accuracy", f"{accuracy}%")

st.markdown("---")
# =====================================
# DIABETES PREDICTION
# =====================================

if predict:

    # Patient Input
    patient_data = np.array([[preg, glucose, bp, skin,
                              insulin, bmi, dpf, age]])

    # Scale Input
    patient_scaled = scaler.transform(patient_data)

    # Prediction
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0][1] * 100

    st.header("🩺 Prediction Result")

    left, right = st.columns(2)

    with left:

        if prediction == 1:
            st.error("🔴 High Risk of Diabetes")
        else:
            st.success("🟢 Low Risk (Non-Diabetic)")

        st.metric(
            "Prediction Probability",
            f"{probability:.2f}%"
        )

    with right:

        st.subheader("Risk Meter")

        st.progress(min(int(probability), 100))

        if probability >= 70:
            st.error("Risk Level : HIGH")
        elif probability >= 40:
            st.warning("Risk Level : MODERATE")
        else:
            st.success("Risk Level : LOW")

    st.markdown("---")

    # =====================================
    # PATIENT SUMMARY
    # =====================================

    st.subheader("📋 Patient Details")

    summary = pd.DataFrame({
        "Parameter": [
            "Pregnancies",
            "Glucose",
            "Blood Pressure",
            "Skin Thickness",
            "Insulin",
            "BMI",
            "Diabetes Pedigree Function",
            "Age"
        ],
        "Value": [
            preg,
            glucose,
            bp,
            skin,
            insulin,
            bmi,
            dpf,
            age
        ]
    })

    st.dataframe(summary, use_container_width=True)

    st.markdown("---")

    # =====================================
    # HEALTH RECOMMENDATION
    # =====================================

    st.subheader("💡 Health Recommendation")

    if probability >= 70:

        st.error("⚠ High Diabetes Risk")

        st.write("✔ Consult a diabetologist immediately.")
        st.write("✔ Monitor blood glucose regularly.")
        st.write("✔ Reduce sugar and sweetened drinks.")
        st.write("✔ Follow a diabetic diet.")
        st.write("✔ Exercise at least 30 minutes daily.")
        st.write("✔ Maintain a healthy body weight.")

    elif probability >= 40:

        st.warning("⚠ Moderate Diabetes Risk")

        st.write("✔ Exercise regularly.")
        st.write("✔ Avoid junk food.")
        st.write("✔ Eat more vegetables and fruits.")
        st.write("✔ Maintain healthy body weight.")
        st.write("✔ Check blood sugar every 6 months.")

    else:

        st.success("✅ Low Diabetes Risk")

        st.write("✔ Continue healthy eating.")
        st.write("✔ Exercise regularly.")
        st.write("✔ Drink enough water.")
        st.write("✔ Sleep 7–8 hours daily.")
        st.write("✔ Get regular health check-ups.")

    st.markdown("---")
    # =====================================
# DATA VISUALIZATION
# =====================================

st.header("📊 Diabetes Data Visualization")

# Create readable labels
chart_df = df.copy()
chart_df["Result"] = chart_df["Outcome"].replace({
    0: "Non-Diabetic",
    1: "Diabetic"
})

# -----------------------------
# Pie Chart & Glucose Histogram
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    fig1 = px.pie(
        chart_df,
        names="Result",
        title="Diabetic vs Non-Diabetic Patients",
        hole=0.45
    )

    st.plotly_chart(fig1, use_container_width=True)

with col2:

    fig2 = px.histogram(
        chart_df,
        x="Glucose",
        color="Result",
        title="Glucose Distribution",
        barmode="overlay"
    )

    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# BMI Histogram & Scatter Plot
# -----------------------------

col3, col4 = st.columns(2)

with col3:

    fig3 = px.histogram(
        chart_df,
        x="BMI",
        color="Result",
        title="BMI Distribution",
        barmode="overlay"
    )

    st.plotly_chart(fig3, use_container_width=True)

with col4:

    fig4 = px.scatter(
        chart_df,
        x="Glucose",
        y="BMI",
        color="Result",
        size="Age",
        hover_data=["BloodPressure"],
        title="Glucose vs BMI"
    )

    st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# Age Box Plot & Insulin Scatter
# -----------------------------

col5, col6 = st.columns(2)

with col5:

    fig5 = px.box(
        chart_df,
        x="Result",
        y="Age",
        color="Result",
        title="Age Distribution"
    )

    st.plotly_chart(fig5, use_container_width=True)

with col6:

    fig6 = px.scatter(
        chart_df,
        x="Age",
        y="Insulin",
        color="Result",
        size="BMI",
        title="Age vs Insulin"
    )

    st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# =====================================
# DATASET PREVIEW
# =====================================

st.subheader("📋 Dataset Preview")

st.dataframe(chart_df.head(10), use_container_width=True)

st.markdown("---")

# =====================================
# FEATURE STATISTICS
# =====================================

st.subheader("📈 Dataset Statistics")

st.dataframe(df.describe(), use_container_width=True)

st.markdown("---")

# =====================================
# FEATURE CORRELATION
# =====================================

st.subheader("📊 Correlation Matrix")

corr = df.corr(numeric_only=True)

fig7 = px.imshow(
    corr,
    text_auto=True,
    title="Correlation Heatmap",
    aspect="auto"
)

st.plotly_chart(fig7, use_container_width=True)

st.markdown("---")

# =====================================
# FOOTER
# =====================================

st.success("✅ Intelligent Diabetes Prediction System Completed Successfully")

st.caption("Developed using Python, Streamlit, Scikit-learn and Plotly")

