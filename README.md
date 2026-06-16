# 🫀 PulseMetrix

PulseMetrix is a production-grade machine learning application designed to predict the 10-year risk of Coronary Heart Disease (CHD). Powered by a refined clinical pipeline trained on the historic **Framingham Heart Study** dataset, this project replaces rigid, technical medical interfaces with an intuitive **Main-Page Control Matrix** and family-friendly visual tracking.

---

## 📈 Model Training Performance Summary

The underlying pipeline evaluates both a regularized **Logistic Regression** model and a **Gradient Boosting Classifier** challenger. 

### 🏆 Preferred Model Selection
**Logistic Regression** was selected as the final production model over Gradient Boosting. While tree-based ensembles like Gradient Boosting are highly capable of capturing complex non-linear combinations, they heavily overfit the minority class on this specific tabular dataset, even when balanced using SMOTE. 

Logistic Regression—bolstered by engineered linear interaction terms—demonstrated superior generalization and stability, achieving a significantly higher test ROC-AUC.

### ⚙️ Pipeline Configurations
* **Selected Model:** Logistic Regression (`C=0.01`, `penalty='l2'`, `solver='liblinear'`)
* **Class Imbalance Strategy:** SMOTE (Synthetic Minority Over-sampling Technique)
* **Cross-Validation Scoring:** Stratified 5-Fold CV (`AUC: 0.7481`)
* **Final Evaluation Metric:** Test ROC-AUC of **`0.7018`** *(Outperformed Gradient Boosting's `0.6364`)*

### 🔍 Clinical Threshold Tuning (0.50 vs 0.30)
In medical diagnostics, missing a true high-risk patient (False Negative) is significantly more dangerous than flagging a healthy patient for extra testing (False Positive). Therefore, the system incorporates a **tuned clinical threshold of 0.30** to optimize for **Recall**:

| Metric (CHD Class) | Standard Threshold (0.50) | Tuned Threshold (0.30) | Clinical Impact |
| :--- | :---: | :---: | :--- |
| **Recall (Sensitivity)** | 63% | **91%** | **Catches 9 out of 10** actual CHD cases. |
| **Precision** | 26% | 20% | Trade-off: Higher rate of follow-up validation required. |
| **Overall Accuracy** | 66.8% | 41.6% | Lowered globally to aggressively defensive safety settings. |

#### Tuned Threshold Confusion Matrix
* **True Negatives (Correctly identified healthy):** 260
* **False Positives (Flagged for clinical review):** 535
* **False Negatives (Missed cases):** **Only 13**
* **True Positives (Correctly flagged high-risk):** 130

---

## 🛠️ Engineered Features

To improve linear predictability, the training script constructs **11 mathematical features** from the raw inputs prior to scaling:
1. `pulse_pressure`: $Systolic\ BP - Diastolic\ BP$ (Key cardiovascular stiffness marker)
2. `MAP`: Mean Arterial Pressure $\rightarrow Diastolic\ BP + \frac{Pulse\ Pressure}{3}$
3. `chol_glucose_ratio`: Ratio of Total Cholesterol to Fasting Glucose
4. `age_x_cigs`: Compounding interaction feature between Age and Smoking frequency

---

## 🖥️ Streamlit App Walkthrough & Interface Guide

This section explains how to navigate the interactive dashboard and outlines the technical operations happening behind the scenes.

### 📥 1. Patient Structural Profile Matrix (Main Page)
Unlike traditional applications that crowd the sidebar, PulseMetrix utilizes an inline configuration grid spread across **three clean container cards at the top of the main layout**:
* **👤 Core Demographics:** Age slider and Biological Sex.
* **🚬 Behavioral Indicators:** Interactive smoking switch toggles. If **"Current Active Smoker"** is disabled, the daily cigarette slider programmatically locks out, automatically passing a clean `0` value to prevent input noise. It also aggregates clinical background checks (BP Meds, Stroke history, Hypertension, Diabetes).
* **🧪 Lab Quantified Vitals & Assays:** Direct numeric metric entry fields for Total Cholesterol, Blood Pressure (Systolic & Diastolic bounds), BMI, Heart Rate, and Fasting Glucose.
![Profile Matrix](assets/app_overview.png)

---

### 🗂️ 2. Tabbed Architecture & Multi-Generational UI

The interface processes and displays predictions across three intuitive sections, rewritten specifically using a conversational, gentle tone accessible to children and elderly patients alike.

#### Tab 1: 📊 Risk Assessment
This tab handles primary pipeline execution. Clicking **"Run Predictive Diagnostics"** standardizes vectors using `scaler.pkl` and runs the tuned Logistic Regression binary.
* **Visuals:** A Plotly gauge chart dynamically shifts colors based on risk severity (Green for All Good, Orange for Gentle Warning, Red for Time to Take Care) against the custom 0.30 boundary line.
* **Conversational Feedback Loops:** * *Low Risk:* Reassures users with bright, supportive language (*"Keep playing, walking, eating your greens, and enjoying your daily routines to keep your heart smiling!"*).
  * *High Risk:* Avoids cold or terrifying technical alerts, phrasing findings gently (*"Your heart score is quite high today. This just means your body is asking for some extra attention."*) and guiding them to share results with a loving family member or friend to plan a medical visit.
![Risk Assessment](assets/tab1_risk_assessment.png)


#### Tab 2: 📈 Health Metrics
Designed for patient feedback and engineering audit logs.
* **Comparative Bar Chart:** An interactive Plotly bar chart visually contrasts the patient's metrics against standard normal population thresholds.
* **Normalized Radar Chart:** Uses a custom multi-axis radar layout to map out exactly where the user sits relative to a healthy benchmark profile line.
* **Model Insights Expander:** Allows engineers to view the static `roc_curve.png` and `feature_importance.png` files generated during model training right inside the interface.
![Health Metrics](assets/tab2_health_metrics.png)


#### Tab 3: 💡 Recommendations
A rule-based medical safety engine that interprets prediction outputs and inputs to generate clear lifestyle instructions.
* Generates tailored, simplified health recommendations based on active risk factors (e.g., smoking cessation, lipid optimization, or sodium reduction guidelines).
* Provides explicit, clear escalation paths reminding users when to seek immediate medical attention or regular checkups.
![Recommendation](assets/tab3_recommendations.png)



---

## 🛠️ How the App Works: Under the Hood

When a user triggers the diagnostics, data moves through a strict 4-stage pipeline before rendering on the screen:

```text
[1. User Input Raw Matrix Configuration]
                  │
                  ▼
[2. Real-Time Feature Engineering Transforms]
    • pulse_pressure = sys_bp - dia_bp
    • MAP = dia_bp + (pulse_pressure / 3)
    • chol_glucose_ratio = tot_chol / (glucose + 1)
    • age_x_cigs = age * cigs_per_day
                  │
                  ▼
[3. Standard Z-Score Vector Scaling via scaler.pkl]
    • Z = (x - mean) / std_dev
                  │
                  ▼
[4. Model Sigmoid Array Mapping via cardio_guard_model.pkl]
    • Computes probability P(CHD)
    • If P(CHD) >= 0.30 ──> Trigger Conversational High-Risk Alert
```

## Project Structure
```text
├── assets/                     # App screenshots linked inside this README
│   ├── app_overview.png
│   ├── tab1_risk_assessment.png
│   ├── tab2_health_metrics.png
│   └── tab3_recommendations.png
├── models/
│   ├── cardio_guard_model.pkl  # Wrapped dict containing model, features & threshold
│   ├── scaler.pkl              # Fitted StandardScaler instance
│   ├── training_report.txt     # Complete classification reports
│   ├── roc_curve.png           # Visual performance curve
│   └── feature_importance.png  # Absolute coefficients plot
├── app.py                      # Clean Streamlit user interface (3-tab deployment)
├── train_model.py              # Training pipeline with SMOTE & Hyperparameter tuning
├── framingham.csv              # Source dataset (3,751 rows post-cleaning)
└── README.md                   # Project documentation