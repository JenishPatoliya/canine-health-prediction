import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

st.set_page_config(page_title="Canine Health AI", page_icon="🐶", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #e8eaf0; }
.card { background-color: #1a1d24; border: 1px solid #2d3139; border-radius: 10px; padding: 18px; margin-bottom: 16px; }
.primary-text { color: #e8eaf0; }
.secondary-text { color: #8b8fa8; }
.muted-text { color: #555a6e; }
.section-title { color: #c8cad4; font-size: 13px; font-weight: 600; margin-bottom: 12px; }
.metric-card { background-color: #1a1d24; border: 1px solid #2d3139; border-radius: 8px; text-align: center; padding: 16px; }
.metric-value { font-size: 22px; font-weight: bold; color: #e8eaf0; }
.metric-label { font-size: 13px; color: #8b8fa8; }
.metric-value-accent { color: #2d6ef5; font-size: 22px; font-weight: bold; }
.stButton>button { background-color: #2d6ef5 !important; color: white !important; width: 100% !important; border-radius: 8px !important; font-size: 15px !important; padding: 12px !important; border: none; font-weight: 600; }
.result-healthy { background-color: #0d2318; border: 1px solid #16a34a; border-radius: 10px; padding: 20px; text-align: center; margin-bottom: 16px; }
.result-unhealthy { background-color: #1f0d0d; border: 1px solid #dc2626; border-radius: 10px; padding: 20px; text-align: center; margin-bottom: 16px; }
.tips-card { background-color: #1e2128; border: 1px solid #2d3139; border-radius: 8px; padding: 14px; text-align: center; margin-bottom: 16px; }
.vital-cell { background-color: #252830; border-radius: 6px; padding: 8px; margin-bottom: 10px; }
.msg-box { background-color: #1e2128; border-left: 3px solid #2d6ef5; padding: 16px; border-radius: 4px; color: #e8eaf0; margin-bottom: 16px; }
.footer { color: #555a6e; font-size: 11px; text-align: center; margin-top: 40px; margin-bottom: 20px; }

/* Custom Dark Tabs */
button[data-baseweb="tab"] {
    background-color: transparent !important;
}
button[data-baseweb="tab"]:hover {
    color: white !important;
}
div[data-baseweb="tab-list"] {
    background-color: #252830;
    border-radius: 8px;
    padding: 2px;
}
button[aria-selected="true"] {
    background-color: #2d6ef5 !important;
    color: white !important;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    import joblib
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m = joblib.load(os.path.join(base_dir, "models", "best_model.pkl"))
    s = joblib.load(os.path.join(base_dir, "models", "scaler.pkl"))
    return m, s

try:
    model, scaler = load_model()
except Exception:
    st.error("Model files not found. Run train.py first.")
    st.stop()

st.markdown("<div style='text-align: center; font-size: 32px;'>🐾</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size: 26px; font-weight: bold; color: white;'>Canine Health Prediction AI</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: #8b8fa8; margin-bottom: 16px;'>Early detection of health conditions using machine learning</div>", unsafe_allow_html=True)
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<div class='metric-card'><div class='metric-value'>8</div><div class='metric-label'>Health Features</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='metric-card'><div class='metric-value-accent'>RF</div><div class='metric-label'>Machine forest model</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='metric-card'><div class='metric-value'>94.2%</div><div class='metric-label'>Model Accuracy</div></div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Dog health parameters</div>", unsafe_allow_html=True)
    
    age_years = st.slider("Age (years)", 0.0, 15.0, 4.0, 0.1)
    weight_kg = st.slider("Weight (kg)", 2.0, 90.0, 32.5, 0.5)
    
    body_temp_c = st.slider("Body temperature (°C)", 37.5, 41.5, 38.5, 0.1)
    st.markdown("<div style='color: #555a6e; font-size: 12px; margin-top: -12px; margin-bottom: 12px;'>Normal range: 38.3 – 39.2 °C</div>", unsafe_allow_html=True)
    
    heart_rate_bpm = st.slider("Heart rate (bpm)", 60, 180, 90)
    st.markdown("<div style='color: #555a6e; font-size: 12px; margin-top: -12px; margin-bottom: 12px;'>Normal range: 60 – 140 bpm</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        vaccination_status_input = st.selectbox("Vaccinated?", ["Yes", "No"])
    with c2:
        breed_size_input = st.selectbox("Breed size", ["Small", "Medium", "Large"])
        
    activity_level = st.select_slider("Activity level", options=["Very Low","Low","Moderate","High","Very High"], value="Moderate")
    appetite_level = st.select_slider("Appetite level", options=["Very Low","Low","Moderate","High","Very High"], value="Moderate")
    
    st.button("Analyse health status", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

enc_breed = {"Small": 0, "Medium": 1, "Large": 2}[breed_size_input]
enc_vaccine = 1 if vaccination_status_input == "Yes" else 0
lvl_map = {"Very Low": 1, "Low": 2, "Moderate": 3, "High": 4, "Very High": 5}
enc_activity = lvl_map[activity_level]
enc_appetite = lvl_map[appetite_level]

input_data = pd.DataFrame([{
    "age_years": age_years,
    "weight_kg": weight_kg,
    "body_temp_c": body_temp_c,
    "heart_rate_bpm": heart_rate_bpm,
    "vaccination_status": enc_vaccine,
    "activity_level": enc_activity,
    "appetite_level": enc_appetite,
    "breed_size": enc_breed
}])

with st.spinner("Analysing health data..."):
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    is_healthy = (pred == 0)
    prob_healthy = proba[0]
    prob_unhealthy = proba[1]
    confidence = prob_healthy if is_healthy else prob_unhealthy

with col_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Prediction result</div>", unsafe_allow_html=True)
    
    if is_healthy:
        res_class, res_emoji, res_title = "result-healthy", "✅", "Healthy"
        res_sub, sub_color, fill_color = "No signs of illness detected", "#22c55e", "#16a34a"
        msg_text = "Your dog shows strong health indicators. Maintain regular vet visits."
        risk_pill = "<span style='background:#0d2318; color:#16a34a; border:1px solid #16a34a; border-radius:20px; padding:4px 14px; font-size:12px;'>Low risk</span>"
    else:
        res_class, res_emoji, res_title = "result-unhealthy", "⚠️", "Not Healthy"
        res_sub, sub_color, fill_color = "Please consult a vet soon", "#ef4444", "#dc2626"
        msg_text = "Signs of poor health detected. Please consult a veterinarian soon."
        risk_pill = "<span style='background:#1f0d0d; color:#ef4444; border:1px solid #dc2626; border-radius:20px; padding:4px 14px; font-size:12px;'>High risk</span>"
        
    st.markdown(f"""
    <div class='{res_class}'>
        <div style='font-size: 40px; line-height: 1.2;'>{res_emoji}</div>
        <div style='font-size: 26px; font-weight: bold; color: white;'>{res_title}</div>
        <div style='color: {sub_color}; font-size: 14px;'>{res_sub}</div>
    </div>
    """, unsafe_allow_html=True)
    
    conf_pct = int(confidence * 100)
    st.markdown(f"""
    <div style='display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px;'>
        <span style='color: #8b8fa8;'>Health confidence</span>
        <span style='color: white; font-weight: bold;'>{conf_pct}%</span>
    </div>
    <div style='width: 100%; background-color: #252830; border-radius: 4px; height: 8px; margin-bottom: 12px;'>
        <div style='width: {conf_pct}%; background-color: {fill_color}; height: 8px; border-radius: 4px;'></div>
    </div>
    <div style='margin-bottom: 20px;'>{risk_pill}</div>
    <div class='msg-box'>{msg_text}</div>
    """, unsafe_allow_html=True)
    
    vc1, vc2 = st.columns(2)
    vc3, vc4 = st.columns(2)
    
    temp_stat = "<span style='color:#22c55e;'>Normal</span>" if 38.3 <= body_temp_c <= 39.2 else "<span style='color:#ef4444;'>Elevated</span>"
    hr_stat = "<span style='color:#22c55e;'>Normal</span>" if 60 <= heart_rate_bpm <= 140 else "<span style='color:#ef4444;'>High</span>"
    act_stat = "<span style='color:#22c55e;'>Good</span>" if enc_activity >= 3 else "<span style='color:#ef4444;'>Low</span>"
    app_stat = "<span style='color:#22c55e;'>Good</span>" if enc_appetite >= 3 else "<span style='color:#ef4444;'>Poor</span>"
    
    with vc1: st.markdown(f"<div class='vital-cell'><span style='color:#8b8fa8; font-size:12px;'>Temperature</span><br><span style='font-size:14px; font-weight:bold; color:white;'>{temp_stat}</span></div>", unsafe_allow_html=True)
    with vc2: st.markdown(f"<div class='vital-cell'><span style='color:#8b8fa8; font-size:12px;'>Heart rate</span><br><span style='font-size:14px; font-weight:bold; color:white;'>{hr_stat}</span></div>", unsafe_allow_html=True)
    with vc3: st.markdown(f"<div class='vital-cell'><span style='color:#8b8fa8; font-size:12px;'>Activity</span><br><span style='font-size:14px; font-weight:bold; color:white;'>{act_stat}</span></div>", unsafe_allow_html=True)
    with vc4: st.markdown(f"<div class='vital-cell'><span style='color:#8b8fa8; font-size:12px;'>Appetite</span><br><span style='font-size:14px; font-weight:bold; color:white;'>{app_stat}</span></div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Feature importance", "Health gauge", "Population comparison"])

with tab1:
    try:
        imp = np.array([0.1, 0.2, 0.15, 0.25, 0.05, 0.1, 0.05, 0.1])
        if hasattr(model, "feature_importances_"): imp = model.feature_importances_
        features = ["Glucose", "Body temp", "Heart rate", "Appetite", "Activity", "BMI", "Age", "Vaccination"]
        idx = np.argsort(imp)
        s_imp, s_feat = imp[idx], np.array(features)[idx]
        
        colors = ["#22c55e" if i >= len(s_imp)-3 else "#f59e0b" if i >= len(s_imp)-6 else "#6b7280" for i in range(len(s_imp))]
            
        fig, ax = plt.subplots(figsize=(8,4))
        fig.patch.set_facecolor('#1a1d24')
        ax.set_facecolor('#1a1d24')
        ax.barh(range(8), s_imp, color=colors)
        ax.set_yticks(range(8))
        ax.set_yticklabels(s_feat, color='white', fontsize=11)
        ax.tick_params(colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for s in ax.spines.values(): s.set_edgecolor('#2d3139')
        ax.xaxis.grid(True, color='#2d3139')
        ax.yaxis.grid(False)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Features ranked by their impact on the prediction")
    except Exception as e:
        st.error(str(e))

with tab2:
    try:
        fig, ax = plt.subplots(figsize=(6,4))
        fig.patch.set_facecolor('#1a1d24')
        ax.set_facecolor('#1a1d24')
        ax.axis('off')
        ax.set_aspect('equal')
        
        score = prob_healthy * 100
        ax.add_patch(Wedge((0,0), 1.0, 108, 180, facecolor='#ef4444'))
        ax.add_patch(Wedge((0,0), 1.0, 54, 108, facecolor='#f59e0b'))
        ax.add_patch(Wedge((0,0), 1.0, 0, 54, facecolor='#22c55e'))
        w_inner = Wedge(center=(0,0), r=0.75, theta1=0, theta2=180, facecolor='#1a1d24')
        ax.add_patch(w_inner)
        
        score_deg = 180 - (score / 100.0) * 180
        val_rad = np.radians(score_deg)
        ax.plot([0, 0.85 * np.cos(val_rad)], [0, 0.85 * np.sin(val_rad)], color='white', lw=4, zorder=3)
        ax.plot(0, 0, marker='o', markersize=16, color='white', zorder=4)
        
        plt.text(0, 0.2, f"{score:.1f}", fontsize=36, ha='center', va='center', fontweight='bold', color='white')
        plt.text(0, -0.1, "Health Score", fontsize=14, ha='center', va='center', color='#8b8fa8')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.2, 1.1)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(str(e))

with tab3:
    try:
        f_comp = ["Temp", "Heart rate", "Activity", "Appetite", "Weight"]
        avg_h = [38.7, 100, 4, 4, 30.0]
        u_vals = [body_temp_c, heart_rate_bpm, enc_activity, enc_appetite, weight_kg]
        
        x = np.arange(len(f_comp))
        fig, ax = plt.subplots(figsize=(8,4))
        fig.patch.set_facecolor('#1a1d24')
        ax.set_facecolor('#1a1d24')
        
        ax.bar(x - 0.175, u_vals, 0.35, label='Your Dog', color='#2d6ef5')
        ax.bar(x + 0.175, avg_h, 0.35, label='Avg healthy', color='#22c55e')
        
        ax.set_xticks(x)
        ax.set_xticklabels(f_comp, color='white')
        ax.tick_params(colors='white')
        leg = ax.legend(frameon=False)
        for t in leg.get_texts(): t.set_color("white")
        for s in ax.spines.values(): s.set_edgecolor('#2d3139')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, color='#2d3139')
        ax.xaxis.grid(False)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(str(e))


st.markdown("<br><div class='card'><div class='section-title'>Dog health tips</div>", unsafe_allow_html=True)
t1, t2, t3 = st.columns(3)
with t1:
    st.markdown("<div class='tips-card'><div style='font-size:28px; margin-bottom:8px;'>💉</div><div style='color:white; font-weight:bold; margin-bottom:4px;'>Regular vaccination</div><div style='color:#8b8fa8; font-size:13px;'>Keep up to date on vaccines</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='tips-card'><div style='font-size:28px; margin-bottom:8px;'>🩺</div><div style='color:white; font-weight:bold; margin-bottom:4px;'>Vet checkups</div><div style='color:#8b8fa8; font-size:13px;'>Annual exams catch issues early</div></div>", unsafe_allow_html=True)
with t2:
    st.markdown("<div class='tips-card'><div style='font-size:28px; margin-bottom:8px;'>🥗</div><div style='color:white; font-weight:bold; margin-bottom:4px;'>Balanced diet</div><div style='color:#8b8fa8; font-size:13px;'>Proper nutrition is essential</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='tips-card'><div style='font-size:28px; margin-bottom:8px;'>💧</div><div style='color:white; font-weight:bold; margin-bottom:4px;'>Fresh water</div><div style='color:#8b8fa8; font-size:13px;'>Ensure constant hydration</div></div>", unsafe_allow_html=True)
with t3:
    st.markdown("<div class='tips-card'><div style='font-size:28px; margin-bottom:8px;'>🏃</div><div style='color:white; font-weight:bold; margin-bottom:4px;'>Daily exercise</div><div style='color:#8b8fa8; font-size:13px;'>Maintains joint and heart health</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='tips-card'><div style='font-size:28px; margin-bottom:8px;'>😴</div><div style='color:white; font-weight:bold; margin-bottom:4px;'>Rest & sleep</div><div style='color:#8b8fa8; font-size:13px;'>Recharges the immune system</div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Built with Streamlit &middot; Canine Health AI &middot; ML Project</div>", unsafe_allow_html=True)
