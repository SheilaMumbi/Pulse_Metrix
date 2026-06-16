"""
Cardio Guard - Advanced CHD Risk Prediction App
================================================
A production-grade Streamlit app for 10-year Coronary Heart Disease risk prediction.
Built on the Framingham Heart Study dataset with an ensemble model pipeline.
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Cardio Guard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH  = Path(__file__).parent / "models" / "cardio_guard_model.pkl"
SCALER_PATH = Path(__file__).parent / "models" / "scaler.pkl"

# The 7 raw entry fields matching user sidebar selections
FEATURE_COLS = ["age", "cigsPerDay", "totChol", "sysBP", "glucose", "diaBP", "heartRate"]

NORMAL_RANGES = {
    "totChol":   (0,   200,  "< 200 mg/dL is desirable"),
    "sysBP":     (90,  120,  "90–120 mmHg is normal"),
    "diaBP":     (60,  80,   "60–80 mmHg is normal"),
    "glucose":   (70,  100,  "70–100 mg/dL (fasting) is normal"),
    "heartRate": (60,  100,  "60–100 bpm is normal"),
}

RISK_FACTORS_INFO = {
    "age":        "Risk increases significantly after age 45 (men) and 55 (women).",
    "cigsPerDay": "Smoking doubles the risk of heart disease.",
    "totChol":    "High cholesterol leads to plaque buildup in arteries.",
    "sysBP":      "Hypertension strains the heart and damages blood vessels.",
    "diaBP":      "Elevated diastolic pressure increases cardiac workload.",
    "glucose":    "Diabetes is a major independent risk factor for CHD.",
    "heartRate":  "Resting heart rate > 100 bpm is associated with higher risk.",
    "BMI":        "Obesity raises blood pressure, cholesterol, and diabetes risk.",
}

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #c0392b 0%, #8e44ad 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 { font-size: 2.8rem; margin: 0; }
    .main-header p  { font-size: 1.1rem; opacity: 0.9; margin: 0.5rem 0 0; }

    /* Risk result cards */
    .risk-high {
        background: linear-gradient(135deg, #ff4757, #ff6b81);
        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
        font-size: 1.4rem; font-weight: bold; margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 71, 87, 0.4);
    }
    .risk-moderate {
        background: linear-gradient(135deg, #ffa502, #ff6348);
        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
        font-size: 1.4rem; font-weight: bold; margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 165, 2, 0.4);
    }
    .risk-low {
        background: linear-gradient(135deg, #2ed573, #1e90ff);
        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
        font-size: 1.4rem; font-weight: bold; margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(46, 213, 115, 0.4);
    }

    /* Info cards */
    .info-card {
        background: #fff3cd;
        color: #856404;
        border-left: 4px solid #c0392b;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }

    /* Metric card */
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] { background-color: #1a1a2e; }
    section[data-testid="stSidebar"] * { color: #eee !important; }

    .disclaimer-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        color: #856404;
        border-radius: 4px;
        padding: 1rem 1.2rem;
        margin-top: 2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; }
</style>
""", unsafe_allow_html=True)


# ── Helper: load model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    """Load the trained pipeline from disk. Returns (model, scaler) or (None, None)."""
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        artefact = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        if isinstance(artefact, dict) and "model" in artefact:
            model = artefact["model"]
        else:
            model = artefact
            
        return model, scaler
    return None, None


# ── Helper: gauge chart ───────────────────────────────────────────────────────
def make_gauge(probability: float) -> go.Figure:
    pct = probability * 100
    if pct < 20:
        color = "#2ed573"
    elif pct < 40:
        color = "#ffa502"
    else:
        color = "#ff4757"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkgray"},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  20], "color": "#d4efdf"},
                {"range": [20, 40], "color": "#fdebd0"},
                {"range": [40, 100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 30,
            },
        },
        title={"text": "10-Year CHD Risk Probability", "font": {"size": 16}},
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
    return fig


# ── Helper: feature importance radar ─────────────────────────────────────────
def make_radar(user_vals: dict) -> go.Figure:
    """Normalised radar chart showing where the user sits vs healthy ranges."""
    labels, user_norm, healthy_norm = [], [], []

    benchmarks = {
        "Age":         (user_vals["age"],        45,   100),
        "Cigs/Day":    (user_vals["cigsPerDay"],  0,    40),
        "Cholesterol": (user_vals["totChol"],     200,  400),
        "Sys BP":      (user_vals["sysBP"],       120,  200),
        "Glucose":     (user_vals["glucose"],     100,  300),
        "Dia BP":      (user_vals["diaBP"],       80,   160),
        "Heart Rate":  (user_vals["heartRate"],   75,   150),
    }

    for label, (val, healthy_ref, max_val) in benchmarks.items():
        labels.append(label)
        user_norm.append(min(val / max_val, 1.0))
        healthy_norm.append(min(healthy_ref / max_val, 1.0))

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=healthy_norm + [healthy_norm[0]],
        theta=labels + [labels[0]],
        fill="toself", name="Healthy Benchmark",
        line_color="#2ed573", fillcolor="rgba(46,213,115,0.15)"
    ))
    fig.add_trace(go.Scatterpolar(
        r=user_norm + [user_norm[0]],
        theta=labels + [labels[0]],
        fill="toself", name="Your Values",
        line_color="#ff4757", fillcolor="rgba(255,71,87,0.2)"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=350,
        margin=dict(t=30, b=30)
    )
    return fig


# ── Helper: recommendation engine ────────────────────────────────────────────
def generate_recommendations(inputs: dict, prob: float) -> list[str]:
    recs = []
    if inputs["cigsPerDay"] > 0:
        recs.append("🚭 Quit smoking – even cutting down significantly reduces CHD risk within months.")
    if inputs["totChol"] > 200:
        recs.append("🥗 Lower cholesterol – adopt a diet low in saturated fats and speak to your doctor about statins.")
    if inputs["sysBP"] > 130:
        recs.append("💊 Manage blood pressure – aim for < 120/80 mmHg through diet, exercise, and if needed, medication.")
    if inputs["glucose"] > 100:
        recs.append("🍬 Monitor blood sugar – pre-diabetic and diabetic levels significantly raise CHD risk.")
    if inputs["BMI"] > 25:
        recs.append("⚖️ Achieve a healthy weight – losing 5–10% of body weight can lower blood pressure and cholesterol.")
    if inputs["heartRate"] > 100:
        recs.append("🏃 Increase aerobic exercise – aim for 150 min/week of moderate exercise to lower resting heart rate.")
    if prob > 0.3:
        recs.append("🩺 Consult a cardiologist – your risk level warrants a professional cardiovascular assessment.")
    if not recs:
        recs.append("✅ Maintain your healthy lifestyle – keep up with regular exercise, balanced diet, and routine check-ups.")
    return recs


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🫀 Cardio Guard</h1>
        <p>10-Year Coronary Heart Disease Risk Assessment · Framingham Heart Study Dataset</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar – patient profile
    with st.sidebar:
        st.markdown("## 📋 Patient Profile")
        st.markdown("---")

        age         = st.slider("Age (years)",            18, 90, 45)
        sex_label   = st.radio("Biological Sex",          ["Female", "Male"])
        sex_male    = 1 if sex_label == "Male" else 0

        st.markdown("### 🚬 Lifestyle")
        current_smoker = st.checkbox("Current Smoker", value=False)
        cigs_per_day   = st.slider("Cigarettes per Day", 0, 60, 0,
                                   disabled=not current_smoker)
        if not current_smoker:
            cigs_per_day = 0

        st.markdown("### 💊 Medical History")
        bp_meds         = st.checkbox("On Blood Pressure Medication")
        prevalent_stroke = st.checkbox("History of Stroke")
        prevalent_hyp    = st.checkbox("Hypertension Diagnosis")
        diabetes         = st.checkbox("Diabetes Diagnosis")

        st.markdown("### 🩺 Vitals & Labs")
        tot_chol   = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 200, step=5)
        sys_bp     = st.number_input("Systolic BP (mmHg)",        80,  250, 120, step=1)
        dia_bp     = st.number_input("Diastolic BP (mmHg)",       40,  150, 80,  step=1)
        bmi        = st.number_input("BMI",                       10.0, 60.0, 25.0, step=0.1, format="%.1f")
        heart_rate = st.number_input("Heart Rate (bpm)",          40,  150, 75,  step=1)
        glucose    = st.number_input("Fasting Glucose (mg/dL)",   50,  400, 85,  step=1)

        predict_btn = st.button("🔮 Predict CHD Risk", use_container_width=True, type="primary")

    # Collect inputs
    user_inputs = {
        "age": age, "sex_male": sex_male,
        "currentSmoker": int(current_smoker), "cigsPerDay": cigs_per_day,
        "BPMeds": int(bp_meds), "prevalentStroke": int(prevalent_stroke),
        "prevalentHyp": int(prevalent_hyp), "diabetes": int(diabetes),
        "totChol": tot_chol, "sysBP": sys_bp, "diaBP": dia_bp,
        "BMI": bmi, "heartRate": heart_rate, "glucose": glucose,
    }

    # Re-engineer all 4 features inside the prediction logic to perfectly total 11 features
    pulse_pressure = sys_bp - dia_bp
    mean_arterial_pressure = dia_bp + (pulse_pressure / 3)
    chol_glucose_ratio = tot_chol / (glucose + 1)
    age_x_cigs = age * cigs_per_day

    feature_array = np.array([[
        age, cigs_per_day, tot_chol, sys_bp, glucose, dia_bp, heart_rate,
        pulse_pressure, mean_arterial_pressure, chol_glucose_ratio, age_x_cigs
    ]])

    # Load model
    model, scaler = load_model_and_scaler()

    # ── Tabs ───────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        ["📊 Risk Assessment", "📈 Health Metrics", "💡 Recommendations"]
    )

    # Default probability placeholder
    probability = 0.0

    # ── TAB 1: Risk Assessment ────────────────────────────────────────────────
    with tab1:
        if model is None:
            st.warning(
                "⚠️ Trained model not found at `models/cardio_guard_model.pkl`. "
                "Run **`train_model.py`** first to generate the model, then restart the app."
            )
            st.info("👉 While the model loads, explore the other health metrics tabs.")
        else:
            if predict_btn or "last_prob" in st.session_state:
                if predict_btn:
                    scaled = scaler.transform(feature_array)
                    probability = float(model.predict_proba(scaled)[0][1])
                    st.session_state["last_prob"] = probability
                    st.session_state["last_inputs"] = user_inputs
                else:
                    probability = st.session_state["last_prob"]
                    user_inputs = st.session_state["last_inputs"]

                pct = probability * 100

                col_gauge, col_result = st.columns([1, 1])
                with col_gauge:
                    st.plotly_chart(make_gauge(probability), use_container_width=True)

                with col_result:
                    if pct < 20:
                        st.markdown(f'<div class="risk-low">✅ Low Risk &nbsp;·&nbsp; {pct:.1f}%</div>', unsafe_allow_html=True)
                        st.success("Your 10-year CHD risk is **low**. Maintain your healthy habits!")
                    elif pct < 40:
                        st.markdown(f'<div class="risk-moderate">⚠️ Moderate Risk &nbsp;·&nbsp; {pct:.1f}%</div>', unsafe_allow_html=True)
                        st.warning("You have a **moderate** risk. Consider lifestyle changes and speak to your GP.")
                    else:
                        st.markdown(f'<div class="risk-high">🚨 High Risk &nbsp;·&nbsp; {pct:.1f}%</div>', unsafe_allow_html=True)
                        st.error("Your risk is **elevated**. Please consult a cardiologist promptly.")

                    # Key risk flags
                    st.markdown("**Key flags in your profile:**")
                    flags = []
                    if sys_bp > 130: flags.append("🔴 Elevated systolic BP")
                    if tot_chol > 200: flags.append("🔴 High cholesterol")
                    if glucose > 100: flags.append("🔴 Elevated glucose")
                    if cigs_per_day > 0: flags.append("🔴 Active smoker")
                    if bmi > 30: flags.append("🟡 Obese BMI")
                    elif bmi > 25: flags.append("🟡 Overweight BMI")
                    if heart_rate > 100: flags.append("🟡 Elevated heart rate")
                    if not flags:
                        flags = ["🟢 No major risk flags detected"]
                    for f in flags:
                        st.markdown(f"- {f}")

                # Summary table
                st.markdown("---")
                st.markdown("### 📄 Input Summary")
                summary = pd.DataFrame({
                    "Parameter": ["Age", "Sex", "Smoker", "Cigs/Day", "Cholesterol",
                                  "Systolic BP", "Diastolic BP", "BMI", "Heart Rate", "Glucose"],
                    "Your Value": [age, sex_label, "Yes" if current_smoker else "No",
                                   cigs_per_day, f"{tot_chol} mg/dL", f"{sys_bp} mmHg",
                                   f"{dia_bp} mmHg", f"{bmi:.1f}", f"{heart_rate} bpm",
                                   f"{glucose} mg/dL"],
                })
                st.dataframe(summary, use_container_width=True, hide_index=True)
            else:
                st.info("👈 Fill in your health details in the sidebar and click **Predict CHD Risk**.")
                st.markdown("""
                ### About This Tool
                Cardio Guard uses a **logistic regression model** trained on the Framingham Heart Study 
                dataset (3,751 patients) to estimate your 10-year risk of developing Coronary Heart Disease.

                **Features used:**
                - Age, sex, smoking habits
                - Blood pressure (systolic & diastolic)
                - Total cholesterol, glucose, heart rate, BMI
                """)

    # ── TAB 2: Health Metrics ─────────────────────────────────────────────────
    with tab2:
        st.subheader("📈 Your Health Metrics vs Normal Ranges")

        col1, col2 = st.columns(2)

        with col1:
            # Bar chart: vitals vs normals
            metrics = {
                "Cholesterol (mg/dL)": (tot_chol, 200),
                "Systolic BP (mmHg)":  (sys_bp,   120),
                "Diastolic BP (mmHg)": (dia_bp,   80),
                "Glucose (mg/dL)":     (glucose,  100),
                "Heart Rate (bpm)":    (heart_rate, 75),
                "BMI":                 (bmi,      25),
            }
            names  = list(metrics.keys())
            yours  = [v[0] for v in metrics.values()]
            normal = [v[1] for v in metrics.values()]
            colors = ["#ff4757" if y > n else "#2ed573" for y, n in zip(yours, normal)]

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(name="Normal Range", x=names, y=normal,
                                     marker_color="#a8d8ea", opacity=0.6))
            fig_bar.add_trace(go.Bar(name="Your Value", x=names, y=yours,
                                     marker_color=colors, opacity=0.9))
            fig_bar.update_layout(
                barmode="group", title="Your Values vs Normal Thresholds",
                height=380, margin=dict(t=40, b=60),
                legend=dict(orientation="h", y=-0.25)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.plotly_chart(make_radar(user_inputs), use_container_width=True)

        # Normal ranges reference
        st.markdown("### 📚 Reference Ranges")
        ref_data = pd.DataFrame([
            {"Metric": "Total Cholesterol", "Optimal": "< 200 mg/dL", "Borderline": "200–239 mg/dL", "High Risk": "≥ 240 mg/dL"},
            {"Metric": "Systolic BP",       "Optimal": "< 120 mmHg",  "Borderline": "120–139 mmHg",  "High Risk": "≥ 140 mmHg"},
            {"Metric": "Diastolic BP",      "Optimal": "< 80 mmHg",   "Borderline": "80–89 mmHg",    "High Risk": "≥ 90 mmHg"},
            {"Metric": "Fasting Glucose",   "Optimal": "< 100 mg/dL", "Borderline": "100–125 mg/dL", "High Risk": "≥ 126 mg/dL"},
            {"Metric": "BMI",               "Optimal": "18.5–24.9",   "Borderline": "25–29.9",        "High Risk": "≥ 30"},
            {"Metric": "Resting Heart Rate","Optimal": "60–75 bpm",   "Borderline": "76–99 bpm",      "High Risk": "≥ 100 bpm"},
        ])
        st.dataframe(ref_data, use_container_width=True, hide_index=True)

    # ── TAB 3: Recommendations ────────────────────────────────────────────────
    with tab3:
        st.subheader("💡 Personalised Recommendations")
        prob_for_recs = st.session_state.get("last_prob", 0.0)
        recs = generate_recommendations(user_inputs, prob_for_recs)
        for rec in recs:
            st.markdown(f'<div class="info-card">{rec}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🏥 When to See a Doctor")
        st.markdown("""
        Seek **immediate medical attention** if you experience:
        - Chest pain or pressure
        - Shortness of breath at rest
        - Pain radiating to the arm, jaw, or back
        - Sudden dizziness or cold sweats
        
        **Schedule a routine appointment** if your risk score is above 20% or you have 
        multiple risk factors flagged above.
        """)

        st.markdown("---")
        st.markdown("### ℹ️ About the Risk Factors")
        for factor, info in RISK_FACTORS_INFO.items():
            clean_name = (factor.replace("cigsPerDay", "Smoking")
                                .replace("totChol", "Cholesterol")
                                .replace("sysBP", "Systolic BP")
                                .replace("diaBP", "Diastolic BP")
                                .replace("heartRate", "Heart Rate"))
            with st.expander(clean_name):
                st.write(info)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>Medical Disclaimer:</strong> Cardio Guard is an educational tool based on statistical 
        modelling. It does <em>not</em> replace professional medical advice, diagnosis, or treatment. 
        Always consult a qualified healthcare provider regarding any medical condition.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()