"""
PulseMetrix - Advanced CHD Risk Prediction Dashboard
===================================================
A production-grade Streamlit app for 10-year Coronary Heart Disease risk prediction.
Built on the Framingham Heart Study dataset with an optimized pipeline architecture.
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="PulseMetrix",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants & File Paths ──────────────────────────────────────────────────
MODEL_PATH  = Path(__file__).parent / "models" / "cardio_guard_model.pkl"
SCALER_PATH = Path(__file__).parent / "models" / "scaler.pkl"

RISK_FACTORS_INFO = {
    "Age":           "Risk increases significantly after age 45 (men) and 55 (women).",
    "Smoking":       "Smoking doubles the risk of heart disease by introducing chronic vascular oxidative stress.",
    "Cholesterol":   "High cholesterol leads to plaque buildup in major arterial channels.",
    "Systolic BP":   "Hypertension strains the heart muscle and actively damages fine blood vessels.",
    "Diastolic BP":  "Elevated diastolic pressure increases long-term resting cardiac workload.",
    "Glucose":       "Diabetes serves as a major, highly aggressive independent risk factor for CHD.",
    "Heart Rate":    "Resting heart rates consistently over 100 bpm are associated with heightened risk profiles.",
    "BMI":           "Obesity shifts blood pressure bounds, cholesterol configurations, and insulin affinity.",
}

# ── Custom Unified CSS Styling ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header banner */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { font-size: 2.6rem; margin: 0; font-weight: 700; color: #f8fafc; }
    .main-header p  { font-size: 1.1rem; opacity: 0.9; margin: 0.5rem 0 0; color: #cbd5e1; }

    /* Risk result state cards */
    .risk-high {
        background: linear-gradient(135deg, #ff4757, #ff6b81);
        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
        font-size: 1.4rem; font-weight: bold; margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 71, 87, 0.3);
    }
    .risk-moderate {
        background: linear-gradient(135deg, #ffa502, #ff6348);
        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
        font-size: 1.4rem; font-weight: bold; margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 165, 2, 0.3);
    }
    .risk-low {
        background: linear-gradient(135deg, #2ed573, #1e90ff);
        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
        font-size: 1.4rem; font-weight: bold; margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(46, 213, 115, 0.3);
    }

    /* Recommendation Info elements */
    .info-card {
        background: #f8fafc;
        color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.6rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .disclaimer-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        color: #856404;
        border-radius: 4px;
        padding: 1rem 1.2rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: load model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    """Load the trained pipeline artifacts safely from disk."""
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        artefact = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        if isinstance(artefact, dict) and "model" in artefact:
            model = artefact["model"]
            threshold = artefact.get("threshold", 0.30)
            features = artefact.get("features", None)
        else:
            model = artefact
            threshold = 0.30
            features = None
            
        return model, scaler, threshold, features
    return None, None, 0.30, None


# ── Helper: gauge chart ───────────────────────────────────────────────────────
def make_gauge(probability: float, threshold: float) -> go.Figure:
    pct = probability * 100
    thresh_pct = threshold * 100
    
    if pct < 15:
        color = "#2ed573"
    elif pct < thresh_pct:
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
                {"range": [0, 15], "color": "#e2e8f0"},
                {"range": [15, thresh_pct], "color": "#fef3c7"},
                {"range": [thresh_pct, 100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": thresh_pct,
            },
        },
        title={"text": "10-Year CHD Risk Probability", "font": {"size": 16}},
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
    return fig


# ── Helper: feature importance radar ─────────────────────────────────────────
def make_radar(user_vals: dict) -> go.Figure:
    """Normalised radar chart mapping user configuration entries to boundaries."""
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
        fill="toself", name="Optimal Bound Baseline",
        line_color="#2ed573", fillcolor="rgba(46,213,115,0.12)"
    ))
    fig.add_trace(go.Scatterpolar(
        r=user_norm + [user_norm[0]],
        theta=labels + [labels[0]],
        fill="toself", name="Subject Vector",
        line_color="#ff4757", fillcolor="rgba(255,71,87,0.18)"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
        showlegend=True, height=350,
        margin=dict(t=30, b=30),
        legend=dict(orientation="h", y=-0.1)
    )
    return fig


# ── Helper: recommendation engine ────────────────────────────────────────────
def generate_recommendations(inputs: dict, prob: float, threshold: float) -> list[str]:
    recs = []
    if inputs["cigsPerDay"] > 0:
        recs.append("🚭 Smoking Cessation Protocol: Complete elimination of daily inhaled nicotine structures to eliminate chronic vascular stress.")
    if inputs["totChol"] > 200:
        recs.append("🥗 Lipid Management Optimization: Pivot macronutrient targets toward high viscous fibers and restrict saturated lipid structures.")
    if inputs["sysBP"] > 130 or inputs["diaBP"] > 80:
        recs.append("💊 Hypertension Mitigation Plan: Restrict trace sodium inputs beneath 1,500mg daily to stabilize plasma volume constraints.")
    if inputs["glucose"] > 100:
        recs.append("🍬 Glycemic Equilibrium Monitoring: Eliminate refined carbohydrate sequences and schedule fasting HbA1c screening.")
    if inputs["BMI"] > 25:
        recs.append("⚖️ Metabolic Weight Alignment: Aim for a target reduction of 5–10% of total body mass index to reduce overall cardiac load.")
    if inputs["heartRate"] > 100:
        recs.append("🏃 Aerobic Conditioning Integration: Implement 150 minutes of weekly moderate aerobic activities to improve resting stroke volume.")
    if prob >= threshold:
        recs.append("🩺 Cardiology Specialist Referral: The computed profile suggests scheduling advanced diagnostic reviews (such as exercise stress tracking).")
    if not recs:
        recs.append("✅ Stable Configuration Maintained: Continue maintaining current lifestyle baselines, dietary tracking, and scheduled routine evaluations.")
    return recs


# ── Main App Execution ────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🫀 PulseMetrix Diagnostic Panel</h1>
        <p>Production-Grade 10-Year Coronary Heart Disease Machine Learning Prediction Engine</p>
    </div>
    """, unsafe_allow_html=True)

    # Load machine learning binaries
    model, scaler, clinical_threshold, model_features = load_model_and_scaler()

    if model is None or scaler is None:
        st.error("🚨 Critical Execution Error: Core diagnostic binaries not found in `models/`. Run **`train_model.py`** prior to launching the dashboard server.")
        st.stop()

    # ── NEW MAIN DASHBOARD PATIENT ENTRY CONTROL MATRIX (NO SIDEBAR) ─────────
    with st.container(border=True):
        st.markdown("### 📋 Patient Structural Profile Matrix")
        
        col_demo, col_life, col_vitals = st.columns(3)

        with col_demo:
            st.markdown("**👤 Core Demographics**")
            age = st.slider("Age (Years)", min_value=18, max_value=90, value=45, step=1)
            sex_label = st.selectbox("Biological Sex at Birth", options=["Female", "Male"], index=0)
            sex_male = 1 if sex_label == "Male" else 0

        with col_life:
            st.markdown("**🚬 Behavioral Indicators**")
            current_smoker = st.toggle("Current Active Smoker", value=False)
            
            if current_smoker:
                cigs_per_day = st.slider("Cigarettes per Day", min_value=1, max_value=60, value=15)
            else:
                st.caption("🔒 Cigarette slider locked to 0 (Subject declared non-smoker)")
                cigs_per_day = 0

            # Structural secondary background inputs
            st.markdown("**📋 Clinical Background Context**")
            sub_c1, sub_c2 = st.columns(2)
            with sub_c1:
                bp_meds = st.checkbox("On BP Meds")
                prevalent_stroke = st.checkbox("History of Stroke")
            with sub_c2:
                prevalent_hyp = st.checkbox("Hypertension Diagnosed")
                diabetes = st.checkbox("Diabetes Diagnosed")

        with col_vitals:
            st.markdown("**🧪 Lab Quantified Vitals & Assays**")
            v_sub1, v_sub2 = st.columns(2)
            with v_sub1:
                tot_chol = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 200, step=5)
                sys_bp   = st.number_input("Systolic BP (mmHg)",        80, 250, 120, step=1)
                dia_bp   = st.number_input("Diastolic BP (mmHg)",       40, 150, 80,  step=1)
            with v_sub2:
                bmi        = st.number_input("Body Mass Index (BMI)", 10.0, 60.0, 25.0, step=0.1, format="%.1f")
                heart_rate = st.number_input("Heart Rate (bpm)",       40, 150, 75,  step=1)
                glucose    = st.number_input("Fasting Glucose (mg/dL)", 50, 400, 85,  step=1)

        st.markdown("---")
        btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 2])
        with btn_col2:
            predict_btn = st.button("📊 Run Predictive Diagnostics", use_container_width=True, type="primary")

    # Mapping vector properties
    user_inputs = {
        "age": age, "sex_male": sex_male,
        "currentSmoker": int(current_smoker), "cigsPerDay": cigs_per_day,
        "BPMeds": int(bp_meds), "prevalentStroke": int(prevalent_stroke),
        "prevalentHyp": int(prevalent_hyp), "diabetes": int(diabetes),
        "totChol": tot_chol, "sysBP": sys_bp, "diaBP": dia_bp,
        "BMI": bmi, "heartRate": heart_rate, "glucose": glucose,
    }

    # ── REAL-TIME FEATURE ENGINEERING TRANSFORMATIONS ───────────────────────
    pulse_pressure = sys_bp - dia_bp
    mean_arterial_pressure = dia_bp + (pulse_pressure / 3)
    chol_glucose_ratio = tot_chol / (glucose + 1)
    age_x_cigs = age * cigs_per_day

    # Dictionary containing all possible engineered dimensions
    full_feature_map = {
        "male": sex_male, "age": age, "currentSmoker": int(current_smoker), "cigsPerDay": cigs_per_day,
        "totChol": tot_chol, "sysBP": sys_bp, "diaBP": dia_bp, "BMI": bmi, "heartRate": heart_rate, "glucose": glucose,
        "pulse_pressure": pulse_pressure, "MAP": mean_arterial_pressure, 
        "chol_glucose_ratio": chol_glucose_ratio, "age_x_cigs": age_x_cigs
    }

    # If the wrapped model package didn't specify features, default back to historical index orders
    if model_features is None:
        feature_array = np.array([[
            age, cigs_per_day, tot_chol, sys_bp, glucose, dia_bp, heart_rate,
            pulse_pressure, mean_arterial_pressure, chol_glucose_ratio, age_x_cigs
        ]])
    else:
        # Dynamically filter and index map input matching your exact training schema
        feature_array = np.array([[full_feature_map[f] for f in model_features]])

    # ── OUTPUT TABS GENERATION LAYER ─────────────────────────────────────────
    st.markdown("### 🧬 Analysis Matrix Results")
    tab1, tab2, tab3 = st.tabs(["📊 Risk Assessment", "📈 Health Metrics", "💡 Recommendations"])

    probability = 0.0

    # TAB 1: Risk Assessment Evaluation Panel
    with tab1:
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
                st.plotly_chart(make_gauge(probability, clinical_threshold), use_container_width=True)

            with col_result:
                if pct < 15:
                    st.markdown(f'<div class="risk-low">✅ All Good! &nbsp;·&nbsp; {pct:.1f}%</div>', unsafe_allow_html=True)
                    st.success("**Wonderful news!** Your heart score is looking great and healthy. Keep playing, walking, eating your greens, and enjoying your daily routines to keep your heart smiling!")
                elif pct < (clinical_threshold * 100):
                    st.markdown(f'<div class="risk-moderate">⚠️ Gentle Warning &nbsp;·&nbsp; {pct:.1f}%</div>', unsafe_allow_html=True)
                    st.warning("**Let\'s be a bit careful:** Your score is in the middle. There is no need to worry, but it is a good reminder to eat a bit healthier, take nice daily walks, and mention this score to your family doctor the next time you visit them.")
                else:
                    st.markdown(f'<div class="risk-high">🚨 Time to Take Care &nbsp;·&nbsp; {pct:.1f}%</div>', unsafe_allow_html=True)
                    st.error(f"🔴 **Please Read: Time for a Little Extra Care**\n\nYour heart score is quite high today. This just means your body is asking for some extra attention. Please share these results with a loving family member or friend, and make a plan to visit a doctor soon so they can give you the best advice on staying strong.")

                # Risk Factor Anomaly Audit Checks
                st.markdown("**Vitals Anomalies Audit:**")
                flags = []
                if sys_bp >= 130 or dia_bp >= 80: flags.append("🔴 Elevated Blood Pressure Profile (Hypertension Framework)")
                if tot_chol >= 200:               flags.append("🔴 Serum Lipids Outside Optimal Boundary")
                if glucose >= 100:                flags.append("🔴 Fasting Glucose Boundary Deviation")
                if cigs_per_day > 0:              flags.append("🔴 Active Inhaled Nicotine Usage")
                if bmi >= 30:                     flags.append("🟡 High Body Mass Index Indexing")
                elif bmi > 25:                    flags.append("🟡 Overweight BMI Baseline")
                if heart_rate > 100:              flags.append("🟡 Tachycardic Resting Pulse Signature")
                
                if not flags:
                    flags = ["🟢 Profile clean of major outlier risk indicators."]
                for f in flags:
                    st.markdown(f"- {f}")

            # Inline Input Summary Panel
            st.markdown("---")
            st.markdown("### 📄 Active Parameters Matrix Log")
            summary = pd.DataFrame({
                "Parameter Field": ["Subject Age", "Biological Sex", "Active Smoker Indicator", "Cigarette Daily Intake", "Serum Total Cholesterol",
                                  "Systolic Pressure Bound", "Diastolic Pressure Bound", "Body Mass Index", "Resting Beats/Min", "Fasting Plasma Glucose"],
                "Registered Value": [age, sex_label, "Yes" if current_smoker else "No",
                                       cigs_per_day, f"{tot_chol} mg/dL", f"{sys_bp} mmHg",
                                       f"{dia_bp} mmHg", f"{bmi:.1f}", f"{heart_rate} bpm",
                                       f"{glucose} mg/dL"],
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)
        else:
            st.info("💡 Adjust the metric controllers above and trigger **Run Predictive Diagnostics** to generate the probability matrices.")
            st.markdown("""
            ### Engine Core Foundations
            PulseMetrix maps user records directly against a **regularized Logistic Regression pipeline** optimized on the historic Framingham Heart Study tracking matrix (3,751 distinct subjects).
            """)

    # TAB 2: Health Metrics Geometric Alignment Comparison
    with tab2:
        st.subheader("📈 Quantitative Metric Trajectory Comparison")
        col1, col2 = st.columns(2)

        with col1:
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
            fig_bar.add_trace(go.Bar(name="Optimal Bound", x=names, y=normal, marker_color="#cbd5e1", opacity=0.5))
            fig_bar.add_trace(go.Bar(name="Current Vector", x=names, y=yours, marker_color=colors, opacity=0.9))
            fig_bar.update_layout(
                barmode="group", title="Metric Vectors vs Population Baselines",
                height=380, margin=dict(t=40, b=60),
                legend=dict(orientation="h", y=-0.25)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.plotly_chart(make_radar(user_inputs), use_container_width=True)

        # Baseline Threshold Reference Matrices
        st.markdown("### 📚 Reference Guideline Frameworks")
        ref_data = pd.DataFrame([
            {"Metric Feature": "Total Cholesterol", "Optimal Target": "< 200 mg/dL", "Borderline Zone": "200–239 mg/dL", "High Risk Boundary": "≥ 240 mg/dL"},
            {"Metric Feature": "Systolic Pressure", "Optimal Target": "< 120 mmHg",  "Borderline Zone": "120–139 mmHg",  "High Risk Boundary": "≥ 140 mmHg"},
            {"Metric Feature": "Diastolic Pressure", "Optimal Target": "< 80 mmHg",   "Borderline Zone": "80–89 mmHg",    "High Risk Boundary": "≥ 90 mmHg"},
            {"Metric Feature": "Fasting Blood Sugar", "Optimal Target": "< 100 mg/dL", "Borderline Zone": "100–125 mg/dL", "High Risk Boundary": "≥ 126 mg/dL"},
            {"Metric Feature": "Body Mass Index", "Optimal Target": "18.5–24.9",   "Borderline Zone": "25–29.9",        "High Risk Boundary": "≥ 30"},
            {"Metric Feature": "Resting Heart Rate", "Optimal Target": "60–75 bpm",   "Borderline Zone": "76–99 bpm",      "High Risk Boundary": "≥ 100 bpm"},
        ])
        st.dataframe(ref_data, use_container_width=True, hide_index=True)

    # TAB 3: Rule-Based Recommendations Framework Engine
    with tab3:
        st.subheader("💡 Expert Intervention Guidelines")
        prob_for_recs = st.session_state.get("last_prob", 0.0)
        recs = generate_recommendations(user_inputs, prob_for_recs, clinical_threshold)
        for rec in recs:
            st.markdown(f'<div class="info-card">{rec}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🏥 Structural Clinical Escalation Thresholds")
        st.markdown("""
        Seek **emergency evaluation channels immediately** if you observe the following symptoms:
        - Acute thoracic chest discomfort, squeeze, or pressure grids.
        - Unprovoked breathing difficulties or shortness of breath at rest.
        - Radiation of localized pain sequences toward upper arms, jaw line, or left shoulder.
        - Acute unexplained vertigo or sudden cold sweats.
        """)

        st.markdown("---")
        st.markdown("### ℹ️ Dynamic Structural Background Contexts")
        for factor, info in RISK_FACTORS_INFO.items():
            with st.expander(factor):
                st.write(info)

    # Standard Regulatory Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>Medical Data Disclaimer:</strong> PulseMetrix serves purely as an automated analytical and portfolio demonstration system. Calculations reflect mathematical models based on statistical population aggregations and cannot be used as an substitute for direct professional physician consultation, medical screening, or customized medical diagnosis.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()