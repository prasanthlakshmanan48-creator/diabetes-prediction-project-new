import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# =====================================
# SVG ICON SYSTEM (LUCIDE VECTORS)
# =====================================

SVG_ICONS = {
    "stethoscope": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><path d="M4.8 2.3A.3.3 0 0 0 4.5 2.6V7a5 5 0 0 0 10 0V2.6a.3.3 0 0 0-.3-.3"/><path d="M9.5 12a6 6 0 0 0 6 6h1a4 4 0 0 0 4-4V9a.3.3 0 0 0-.3-.3"/><circle cx="20" cy="8" r="2"/></svg>',
    "search": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>',
    "activity": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
    "clipboard": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M12 11h4"/><path d="M12 16h4"/><path d="M8 11h.01"/><path d="M8 16h.01"/></svg>',
    "lightbulb": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg>',
    "shield_alert": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg>',
    "shield_warn": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>',
    "shield_check": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/></svg>',
    "leaf": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>',
    "heart_pulse": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/><path d="M12 5v14"/></svg>',
    "calendar_check": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><path d="M8 2v4"/><path d="M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/><path d="m9 16 2 2 4-4"/></svg>',
    "info": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>',
    "check_circle": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>',
    "database": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></svg>',
    "table": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><path d="M12 3v18"/><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/></svg>',
    "gauge": '<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>'
}

def svg_icon(name, color="currentColor", size=24, stroke=2):
    template = SVG_ICONS.get(name, "")
    return template.format(color=color, size=size, stroke=stroke)

# =====================================
# PAGE CONFIGURATION
# =====================================

PAGE_FAVICON = "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%230f766e' stroke-width='2'><path d='M22 12h-4l-3 9L9 3l-3 9H2'/></svg>"

st.set_page_config(
    page_title="Diabetic Prediction App",
    page_icon=PAGE_FAVICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# CUSTOM HEALTHCARE CSS DESIGN SYSTEM
# =====================================

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1350px;
    }

    /* Custom Header Hero */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        padding: 2.25rem 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
    }

    /* Metric Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px -3px rgba(0, 0, 0, 0.08);
    }

    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.35rem;
    }

    /* Prediction Result Cards */
    .result-card-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 1.75rem;
        color: #991b1b;
        box-shadow: 0 10px 20px -5px rgba(239, 68, 68, 0.15);
    }

    .result-card-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 1.75rem;
        color: #065f46;
        box-shadow: 0 10px 20px -5px rgba(16, 185, 129, 0.15);
    }

    /* Recommendation Category Box */
    .rec-box {
        background: #ffffff;
        border-left: 4px solid #0284c7;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }

    .rec-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #0369a1;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Info Alert Box */
    .info-box {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        color: #0369a1;
        display: flex;
        align-items: center;
        gap: 0.85rem;
        font-size: 0.98rem;
    }

    /* Custom Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35em 0.85em;
        font-size: 0.82rem;
        font-weight: 700;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-high { background-color: #ef4444; color: white; }
    .badge-moderate { background-color: #f59e0b; color: white; }
    .badge-low { background-color: #10b981; color: white; }

    /* Section Heading Layout */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 1rem;
    }

    /* Streamlit Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #e2e8f0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 24px;
        background-color: transparent;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
        color: #64748b;
        font-size: 0.95rem;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0f766e !important;
        border-bottom: 3px solid #0f766e !important;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =====================================
# LOAD MODEL & SCALER & DATASET
# =====================================

@st.cache_resource
def load_ml_components():
    model = joblib.load("diabetes_model (1).pkl")
    scaler = joblib.load("scaler (1).pkl")
    return model, scaler

@st.cache_data
def load_dataset():
    df = pd.read_excel("diabetes (1).csv.xlsx")
    return df

model, scaler = load_ml_components()
df = load_dataset()

# Calculate dataset stats
total_records = len(df)
diabetic_count = int((df["Outcome"] == 1).sum())
non_diabetic_count = int((df["Outcome"] == 0).sum())
prevalence_rate = (diabetic_count / total_records) * 100

# =====================================
# INITIALIZE SESSION STATE
# =====================================

if "has_predicted" not in st.session_state:
    st.session_state["has_predicted"] = False

# =====================================
# SIDEBAR INTAKE FORM
# =====================================

sidebar_header_html = f"""
<div style="font-size: 1.2rem; font-weight: 700; color: #0f172a; display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
    {svg_icon('stethoscope', color='#0f766e', size=24)} Patient Intake Form
</div>
<div style="font-size: 0.85rem; color: #64748b; margin-bottom: 1rem;">Enter physiological indicators below.</div>
"""
st.sidebar.markdown(sidebar_header_html, unsafe_allow_html=True)

with st.sidebar.form(key="patient_intake_form"):
    preg = st.number_input(
        "Pregnancies",
        min_value=0, max_value=20, value=2,
        help="Number of times pregnant"
    )
    glucose = st.number_input(
        "Glucose Level (mg/dL)",
        min_value=0, max_value=250, value=120,
        help="Plasma glucose concentration (2 hours in an oral glucose tolerance test)"
    )
    bp = st.number_input(
        "Blood Pressure (mmHg)",
        min_value=0, max_value=150, value=70,
        help="Diastolic blood pressure"
    )
    skin = st.number_input(
        "Skin Thickness (mm)",
        min_value=0, max_value=100, value=20,
        help="Triceps skin fold thickness"
    )
    insulin = st.number_input(
        "Insulin Level (μU/mL)",
        min_value=0, max_value=900, value=85,
        help="2-Hour serum insulin"
    )
    bmi = st.number_input(
        "Body Mass Index (BMI)",
        min_value=0.0, max_value=70.0, value=28.5, step=0.1,
        help="Body mass index (weight in kg/(height in m)^2)"
    )
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0, max_value=3.0, value=0.35, step=0.01,
        help="Scores likelihood of diabetes based on family history"
    )
    age = st.number_input(
        "Age (Years)",
        min_value=1, max_value=120, value=45,
        help="Patient age"
    )

    submit_button = st.form_submit_button(
        "Calculate Risk Score",
        use_container_width=True
    )

if submit_button:
    patient_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    patient_scaled = scaler.transform(patient_data)
    
    pred_class = model.predict(patient_scaled)[0]
    prob_val = model.predict_proba(patient_scaled)[0][1] * 100
    
    st.session_state["has_predicted"] = True
    st.session_state["pred_class"] = int(pred_class)
    st.session_state["prob_val"] = float(prob_val)
    st.session_state["patient_inputs"] = {
        "Pregnancies": preg,
        "Glucose (mg/dL)": glucose,
        "Blood Pressure (mmHg)": bp,
        "Skin Thickness (mm)": skin,
        "Insulin (μU/mL)": insulin,
        "BMI": bmi,
        "Diabetes Pedigree": dpf,
        "Age": age
    }

# =====================================
# HEADER HERO SECTION
# =====================================

hero_html = f"""
<div class="hero-container">
    <div class="hero-title">
        {svg_icon('stethoscope', color='#14b8a6', size=36, stroke=2.5)}
        Diabetic Prediction App
    </div>
    <div class="hero-subtitle">Advanced Machine Learning Diagnostic Dashboard & Exploratory Health Analytics</div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# =====================================
# MAIN TABBED DASHBOARD LAYOUT
# =====================================

tab_predict, tab_analytics, tab_dataset = st.tabs([
    "Clinical Risk Assessment",
    "Data Analytics & Insights",
    "Dataset & Statistics Explorer"
])

# -------------------------------------
# TAB 1: CLINICAL RISK ASSESSMENT
# -------------------------------------
with tab_predict:
    if st.session_state["has_predicted"]:
        pred_class = st.session_state["pred_class"]
        prob_val = st.session_state["prob_val"]
        inputs = st.session_state["patient_inputs"]
        
        # Risk Badge & Level Calculation
        if prob_val >= 70:
            risk_level = "HIGH"
            badge_class = "badge-high"
            card_class = "result-card-high"
            status_text = "High Risk of Diabetes Detected"
            shield_icon = svg_icon("shield_alert", color="#ffffff", size=16)
        elif prob_val >= 40:
            risk_level = "MODERATE"
            badge_class = "badge-moderate"
            card_class = "result-card-high"
            status_text = "Moderate Diabetes Risk Profile"
            shield_icon = svg_icon("shield_warn", color="#ffffff", size=16)
        else:
            risk_level = "LOW"
            badge_class = "badge-low"
            card_class = "result-card-low"
            status_text = "Low Risk (Non-Diabetic Profile)"
            shield_icon = svg_icon("shield_check", color="#ffffff", size=16)

        # Result Summary Banner
        st.markdown(f"""
        <div class="{card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="badge {badge_class}">{shield_icon} {risk_level} RISK LEVEL</span>
                    <h2 style="margin-top: 0.6rem; margin-bottom: 0.25rem; font-weight: 800;">{status_text}</h2>
                    <p style="margin: 0; opacity: 0.9;">Based on standard machine learning clinical feature classification.</p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">Assessed Probability</div>
                    <div style="font-size: 2.5rem; font-weight: 800;">{prob_val:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visual Progress Meter
        gauge_header = f"""
        <div class="section-header">
            {svg_icon('gauge', color='#0f766e', size=22)} Risk Meter Gauge
        </div>
        """
        st.markdown(gauge_header, unsafe_allow_html=True)
        st.progress(min(int(prob_val), 100))
        st.markdown("<br>", unsafe_allow_html=True)

        # 2-Column Split: Patient Vitals vs Categorized Recommendations
        col_left, col_right = st.columns([1, 1.2], gap="large")

        with col_left:
            params_header = f"""
            <div class="section-header">
                {svg_icon('clipboard', color='#0f766e', size=22)} Patient Parameters
            </div>
            """
            st.markdown(params_header, unsafe_allow_html=True)
            summary_df = pd.DataFrame(
                list(inputs.items()),
                columns=["Parameter", "Clinical Value"]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        with col_right:
            recs_header = f"""
            <div class="section-header">
                {svg_icon('lightbulb', color='#0f766e', size=22)} Tailored Clinical Recommendations
            </div>
            """
            st.markdown(recs_header, unsafe_allow_html=True)
            
            if prob_val >= 70:
                st.markdown(f"""
                <div class="rec-box" style="border-left-color: #ef4444;">
                    <div class="rec-title" style="color: #b91c1c;">
                        {svg_icon('stethoscope', color='#b91c1c', size=18)} Urgent Medical Consultation
                    </div>
                    Consult a specialist diabetologist promptly for comprehensive blood testing (HbA1c & Fasting Glucose).
                </div>
                <div class="rec-box" style="border-left-color: #ef4444;">
                    <div class="rec-title" style="color: #b91c1c;">
                        {svg_icon('leaf', color='#b91c1c', size=18)} Dietary Modifications
                    </div>
                    Eliminate refined sugars and sweetened beverages. Transition to low-glycemic index foods and complex carbohydrates.
                </div>
                <div class="rec-box" style="border-left-color: #ef4444;">
                    <div class="rec-title" style="color: #b91c1c;">
                        {svg_icon('heart_pulse', color='#b91c1c', size=18)} Physical Activity & Weight Control
                    </div>
                    Engage in at least 30 minutes of moderate aerobic exercise 5 days a week.
                </div>
                """, unsafe_allow_html=True)
            elif prob_val >= 40:
                st.markdown(f"""
                <div class="rec-box" style="border-left-color: #f59e0b;">
                    <div class="rec-title" style="color: #b45309;">
                        {svg_icon('calendar_check', color='#b45309', size=18)} Preventive Screening
                    </div>
                    Schedule routine blood glucose evaluation every 6 months to monitor glycemic trend.
                </div>
                <div class="rec-box" style="border-left-color: #f59e0b;">
                    <div class="rec-title" style="color: #b45309;">
                        {svg_icon('leaf', color='#b45309', size=18)} Nutritional Balance
                    </div>
                    Increase dietary fiber intake with fresh vegetables, whole grains, and lean protein.
                </div>
                <div class="rec-box" style="border-left-color: #f59e0b;">
                    <div class="rec-title" style="color: #b45309;">
                        {svg_icon('heart_pulse', color='#b45309', size=18)} Active Lifestyle
                    </div>
                    Maintain continuous weekly physical activity and avoid prolonged sedentary intervals.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="rec-box" style="border-left-color: #10b981;">
                    <div class="rec-title" style="color: #047857;">
                        {svg_icon('check_circle', color='#047857', size=18)} Maintain Healthy Wellness Habits
                    </div>
                    Continue balanced nutrition, regular hydration, and consistent nightly sleep (7–8 hours).
                </div>
                <div class="rec-box" style="border-left-color: #10b981;">
                    <div class="rec-title" style="color: #047857;">
                        {svg_icon('calendar_check', color='#047857', size=18)} Annual Checkup
                    </div>
                    Schedule standard annual health physicals for ongoing health monitoring.
                </div>
                """, unsafe_allow_html=True)

    else:
        info_html = f"""
        <div class="info-box">
            {svg_icon('info', color='#0369a1', size=22)}
            <span>Enter patient clinical details in the sidebar form and click <b>"Calculate Risk Score"</b> to view the diagnostic result and care recommendations.</span>
        </div>
        """
        st.markdown(info_html, unsafe_allow_html=True)

# -------------------------------------
# TAB 2: DATA ANALYTICS & INSIGHTS
# -------------------------------------
with tab_analytics:
    # High-level Dataset Metrics Header Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Records</div>
            <div class="kpi-value">{total_records:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Non-Diabetic</div>
            <div class="kpi-value" style="color: #10b981;">{non_diabetic_count:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Diabetic High Risk</div>
            <div class="kpi-value" style="color: #ef4444;">{diabetic_count:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Prevalence Rate</div>
            <div class="kpi-value" style="color: #0284c7;">{prevalence_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    charts_header = f"""
    <div class="section-header">
        {svg_icon('activity', color='#0f766e', size=22)} Interactive Feature Distributions
    </div>
    """
    st.markdown(charts_header, unsafe_allow_html=True)

    # Plotly Color Palette
    COLOR_MAP = {"Non-Diabetic": "#10b981", "Diabetic": "#ef4444"}
    
    chart_df = df.copy()
    chart_df["Diagnosis"] = chart_df["Outcome"].replace({0: "Non-Diabetic", 1: "Diabetic"})

    # Row 1: Pie Chart & Glucose Histogram
    r1_col1, r1_col2 = st.columns(2)

    with r1_col1:
        fig_pie = px.pie(
            chart_df,
            names="Diagnosis",
            title="Patient Outcome Ratio",
            color="Diagnosis",
            color_discrete_map=COLOR_MAP,
            hole=0.45
        )
        fig_pie.update_layout(margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with r1_col2:
        fig_gluc = px.histogram(
            chart_df,
            x="Glucose",
            color="Diagnosis",
            color_discrete_map=COLOR_MAP,
            title="Glucose Concentration Distribution (mg/dL)",
            barmode="overlay",
            opacity=0.75
        )
        fig_gluc.update_layout(margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_gluc, use_container_width=True)

    # Row 2: BMI Histogram & Glucose vs BMI Scatter
    r2_col1, r2_col2 = st.columns(2)

    with r2_col1:
        fig_bmi = px.histogram(
            chart_df,
            x="BMI",
            color="Diagnosis",
            color_discrete_map=COLOR_MAP,
            title="Body Mass Index (BMI) Distribution",
            barmode="overlay",
            opacity=0.75
        )
        fig_bmi.update_layout(margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_bmi, use_container_width=True)

    with r2_col2:
        fig_scatter1 = px.scatter(
            chart_df,
            x="Glucose",
            y="BMI",
            color="Diagnosis",
            color_discrete_map=COLOR_MAP,
            size="Age",
            hover_data=["BloodPressure"],
            title="Glucose vs BMI Relationship (Size = Age)"
        )
        fig_scatter1.update_layout(margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_scatter1, use_container_width=True)

    # Row 3: Age Boxplot & Correlation Heatmap
    r3_col1, r3_col2 = st.columns(2)

    with r3_col1:
        fig_age_box = px.box(
            chart_df,
            x="Diagnosis",
            y="Age",
            color="Diagnosis",
            color_discrete_map=COLOR_MAP,
            title="Age Distribution Across Outcome Groups"
        )
        fig_age_box.update_layout(margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_age_box, use_container_width=True)

    with r3_col2:
        corr_matrix = df.corr(numeric_only=True)
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="Viridis",
            title="Feature Correlation Matrix Heatmap",
            aspect="auto"
        )
        fig_heatmap.update_layout(margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_heatmap, use_container_width=True)

# -------------------------------------
# TAB 3: DATASET & STATISTICS EXPLORER
# -------------------------------------
with tab_dataset:
    stats_header = f"""
    <div class="section-header">
        {svg_icon('database', color='#0f766e', size=22)} Descriptive Feature Statistics
    </div>
    """
    st.markdown(stats_header, unsafe_allow_html=True)
    st.dataframe(df.describe().T, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    table_header = f"""
    <div class="section-header">
        {svg_icon('table', color='#0f766e', size=22)} Patient Records Dataset Browser
    </div>
    """
    st.markdown(table_header, unsafe_allow_html=True)
    
    # Simple search filter
    filter_option = st.selectbox(
        "Filter by Outcome Status",
        options=["All Patients", "Non-Diabetic Only", "Diabetic Only"]
    )
    
    if filter_option == "Non-Diabetic Only":
        filtered_df = chart_df[chart_df["Outcome"] == 0]
    elif filter_option == "Diabetic Only":
        filtered_df = chart_df[chart_df["Outcome"] == 1]
    else:
        filtered_df = chart_df

    st.caption(f"Showing {len(filtered_df)} patient records")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

# =====================================
# FOOTER
# =====================================

st.markdown("---")
footer_html = f"""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding-top: 0.5rem;">
    {svg_icon('check_circle', color='#10b981', size=18)} 
    <b>Diabetic Prediction App</b> | Developed using Python, Streamlit, Scikit-learn, and Plotly Express.
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
