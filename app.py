"""
==========================================
PROJECT : EXAM SCORE PREDICTOR
AUTHOR  : Pratik
==========================================
Streamlit app that predicts a student's exam score with a trained ML model.
UI: dark / light mode, tabbed charts, grade card, recommendations.
"""

import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Exam Score Predictor", page_icon="🎓", layout="wide")

# ==========================================
# THEMES
# ==========================================
THEMES = {
    "Dark": dict(bg="#0e1420", surface="#161e2e", border="#27324a",
                 text="#e6ebf5", muted="#93a0b8", accent="#7c9bff",
                 accent2="#a78bfa", tint="#1c2740", shadow="none",
                 hero="linear-gradient(120deg,#161e2e,#0e1420)",
                 plotly="plotly_dark",
                 colors=["#7c9bff", "#a78bfa", "#38bdf8", "#34d399", "#fbbf24"]),
    "Light": dict(bg="#eef2f9", surface="#ffffff", border="#d3dbec",
                  text="#0f172a", muted="#4f5d78", accent="#2f55d4",
                  accent2="#6d3fd9", tint="#eaf0ff",
                  shadow="0 1px 2px rgba(15,23,42,.05), 0 8px 24px rgba(47,85,212,.08)",
                  hero="linear-gradient(120deg,#e3ebff 0%,#f3eefe 55%,#ffffff 100%)",
                  plotly="plotly_white",
                  colors=["#2f55d4", "#6d3fd9", "#0ea5e9", "#059669", "#d97706"]),
}

with st.sidebar:
    mode = st.segmented_control("Appearance", ["Light", "Dark"], default="Light",
                                selection_mode="single") or "Light"
T = THEMES[mode]

st.markdown(f"""
<style>
:root {{
  color-scheme: {'dark' if mode == 'Dark' else 'light'};
  --bg:{T['bg']}; --surface:{T['surface']}; --border:{T['border']};
  --text:{T['text']}; --muted:{T['muted']}; --tint:{T['tint']};
  --accent:{T['accent']}; --accent2:{T['accent2']}; --shadow:{T['shadow']};
}}
.stApp, [data-testid="stHeader"] {{ background: var(--bg); }}
[data-testid="stSidebar"] {{ background: var(--surface); border-right:1px solid var(--border); }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li,
.stApp label, .stApp [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] h2 {{ color: var(--text); }}
[data-testid="stCaptionContainer"] * {{ color: var(--muted) !important; }}
.block-container {{ padding-top: 2rem; max-width: 1200px; }}

/* Hero */
.hero {{ padding: 30px 34px; border-radius: 20px; border:1px solid var(--border);
  background: {T['hero']}; box-shadow: var(--shadow); margin-bottom: 20px; }}
.hero h1 {{ margin:0; font-size: 2.1rem; font-weight: 750; letter-spacing:-0.02em; }}
.hero p {{ margin:6px 0 0; color: var(--muted) !important; font-size: 1.05rem; }}

/* Metrics */
[data-testid="stMetric"] {{ background: var(--surface); border:1px solid var(--border);
  padding: 16px 18px; border-radius: 16px; box-shadow: var(--shadow); }}
[data-testid="stMetricLabel"] * {{ color: var(--muted) !important; }}
[data-testid="stMetricValue"] * {{ color: var(--text) !important; font-weight:700; }}

/* Inputs */
[data-baseweb="select"] > div, [data-baseweb="input"] > div {{
  background: var(--surface) !important; border:1px solid var(--border) !important;
  border-radius: 12px !important; box-shadow: var(--shadow); }}
[data-baseweb="select"] div, [data-baseweb="select"] span,
[data-baseweb="select"] svg {{ color: var(--text) !important; fill: var(--text) !important; }}
[data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] ul {{
  background: var(--surface) !important; border:1px solid var(--border); border-radius:12px; }}
[data-baseweb="popover"] li {{ color: var(--text) !important; background: transparent !important; }}
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] {{
  background: var(--tint) !important; }}

/* Sliders */
[data-testid="stSlider"] [role="slider"] {{ background: var(--accent) !important;
  border: 3px solid var(--surface); box-shadow: 0 0 0 1px var(--accent); }}
[data-testid="stSlider"] [data-testid="stThumbValue"] {{ color: var(--accent) !important; font-weight:700; }}
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{ color: var(--muted) !important; }}

/* Segmented control */
[data-testid="stSegmentedControl"] button {{ background: var(--tint); border:1px solid var(--border);
  color: var(--text); }}
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] button[aria-pressed="true"] {{
  background: var(--accent); color:#fff; border-color: var(--accent); }}

/* Progress */
[data-testid="stProgress"] > div > div {{ background: var(--tint) !important; }}
[data-testid="stProgress"] > div > div > div {{
  background: linear-gradient(90deg, var(--accent), var(--accent2)) !important; }}

/* Primary button */
div.stButton > button {{ width:100%; height:56px; border-radius:14px; border:none;
  font-size:1.1rem; font-weight:700; color:#fff !important;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  box-shadow: 0 6px 18px rgba(47,85,212,.28); }}
div.stButton > button p {{ color:#fff !important; }}
div.stButton > button:hover {{ filter: brightness(1.07); }}
div.stButton > button:focus-visible {{ outline:3px solid var(--accent); outline-offset:3px; }}

/* Result + tips */
.result {{ border:1px solid var(--border); background: var(--surface); box-shadow: var(--shadow);
  border-radius:20px; padding:24px 28px; display:flex; gap:28px; align-items:center; margin-bottom:12px; }}
.result .score {{ font-size:3.2rem; font-weight:800; line-height:1; color: var(--text); }}
.result .score small {{ font-size:1.1rem; color: var(--muted); font-weight:500; }}
.badge {{ font-size:1.5rem; font-weight:800; padding:10px 22px; border-radius:12px;
  color:#fff; background: linear-gradient(90deg, var(--accent), var(--accent2)); }}
.tip {{ border:1px solid var(--border); border-left:4px solid var(--accent);
  background: var(--surface); box-shadow: var(--shadow); color: var(--text);
  padding:11px 16px; border-radius:12px; margin-bottom:8px; }}

/* Tabs + chart containers */
.stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{ color: var(--muted); }}
.stTabs [aria-selected="true"] {{ color: var(--accent) !important; font-weight:700; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--accent) !important; }}
[data-testid="stPlotlyChart"] {{ background: var(--surface); border:1px solid var(--border);
  border-radius:16px; padding:8px; box-shadow: var(--shadow); }}
hr {{ border-color: var(--border) !important; }}
@media (max-width: 640px) {{ .result {{ flex-direction:column; align-items:flex-start; }} }}
</style>
""", unsafe_allow_html=True)


# ==========================================
# MODEL
# ==========================================
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_student_performance_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file `best_student_performance_model.pkl` not found. "
             "Place it in the `models/` folder.")
    st.stop()

# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div class="hero">
  <h1>🎓 Exam Score Predictor</h1>
  <p>Estimate a student's exam score from daily habits and background.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR INPUTS
# ==========================================
with st.sidebar:
    st.header("📋 Student habits")
    study_hours_per_day = st.slider("📖 Study hours", 0.0, 12.0, 3.0, 0.5)
    social_media_hours = st.slider("📱 Social media hours", 0.0, 12.0, 2.0, 0.5)
    netflix_hours = st.slider("🎬 Netflix hours", 0.0, 10.0, 1.0, 0.5)
    attendance_percentage = st.slider("🏫 Attendance (%)", 0, 100, 75)
    sleep_hours = st.slider("😴 Sleep hours", 0.0, 12.0, 7.0, 0.5)
    exercise_frequency = st.slider("🏃 Exercise days / week", 0, 7, 3)
    mental_health_rating = st.slider("🧠 Mental health (1–10)", 1, 10, 5)

# ==========================================
# SUMMARY
# ==========================================
st.subheader("Lifestyle summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Study", f"{study_hours_per_day} hrs")
m2.metric("Sleep", f"{sleep_hours} hrs")
m3.metric("Attendance", f"{attendance_percentage}%")
m4.metric("Exercise", f"{exercise_frequency} days")

st.subheader("Background")
c1, c2, c3 = st.columns(3)
with c1:
    part_time_job = st.selectbox("💼 Part-time job", [0, 1],
                                 format_func=lambda x: ["No", "Yes"][x])
    extracurricular_participation = st.selectbox("⚽ Extracurriculars", [0, 1],
                                                 format_func=lambda x: ["No", "Yes"][x])
with c2:
    diet_quality = st.selectbox("🥗 Diet quality", [0, 1, 2],
                                format_func=lambda x: ["Poor", "Fair", "Good"][x])
    internet_quality = st.selectbox("🌐 Internet quality", [0, 1, 2],
                                    format_func=lambda x: ["Poor", "Average", "Good"][x])
with c3:
    parental_education_level = st.selectbox(
        "🎓 Parental education", [0, 1, 2],
        format_func=lambda x: ["High School", "Bachelor", "Master"][x])

# ==========================================
# VALIDATION
# ==========================================
used = study_hours_per_day + social_media_hours + netflix_hours + sleep_hours
if used > 24:
    st.error(f"Study + social media + Netflix + sleep add up to {used:.1f} hrs. "
             "Reduce them to 24 hrs or less.")
    st.stop()

st.write("")
predict = st.button("🚀 Predict exam score", use_container_width=True)

# ==========================================
# PREDICTION
# ==========================================
FEATURES = [
    "study_hours_per_day", "social_media_hours", "netflix_hours",
    "attendance_percentage", "sleep_hours", "exercise_frequency",
    "mental_health_rating", "part_time_job", "diet_quality",
    "parental_education_level", "internet_quality",
    "extracurricular_participation",
]


def grade_for(score):
    if score >= 85:
        return "A+"
    if score >= 70:
        return "A"
    if score >= 50:
        return "B"
    return "C"


def style(fig, height=380):
    fig.update_layout(template=T["plotly"], height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=T["text"]), colorway=T["colors"], margin=dict(t=60, b=20, l=20, r=20))
    return fig


if predict:
    data = pd.DataFrame([[
        study_hours_per_day, social_media_hours, netflix_hours,
        attendance_percentage, sleep_hours, exercise_frequency,
        mental_health_rating, part_time_job, diet_quality,
        parental_education_level, internet_quality,
        extracurricular_participation,
    ]], columns=FEATURES)

    with st.spinner("Analyzing..."):
        time.sleep(1)
        score = float(max(0, min(100, model.predict(data)[0])))
    grade = grade_for(score)

    st.write("")
    st.markdown(f"""
    <div class="result">
      <div class="score">{score:.1f}<small> / 100</small></div>
      <div class="badge">Grade {grade}</div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(int(score))

    # ---- Charts ----
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=score, title={"text": "Exam score"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": T["accent"], "thickness": 0.3},
               "steps": [
                   {"range": [0, 50], "color": "#fca5a5" if mode == "Light" else "#ef4444"},
                   {"range": [50, 70], "color": "#fcd34d" if mode == "Light" else "#f59e0b"},
                   {"range": [70, 85], "color": "#7dd3fc" if mode == "Light" else "#38bdf8"},
                   {"range": [85, 100], "color": "#86efac" if mode == "Light" else "#22c55e"}]}))
    style(gauge)

    pie = px.pie(
        pd.DataFrame({
            "Activity": ["Study", "Social media", "Netflix", "Sleep", "Other"],
            "Hours": [study_hours_per_day, social_media_hours, netflix_hours,
                      sleep_hours, 24 - used]}),
        names="Activity", values="Hours", hole=0.55, title="Daily time distribution")
    style(pie)

    radar_vals = [study_hours_per_day * 8, attendance_percentage, sleep_hours * 8,
                  mental_health_rating * 10, exercise_frequency * 14]
    radar_labels = ["Study", "Attendance", "Sleep", "Mental health", "Exercise"]
    radar = go.Figure(go.Scatterpolar(
        r=radar_vals + radar_vals[:1], theta=radar_labels + radar_labels[:1],
        fill="toself", line=dict(color=T["accent"]), fillcolor=T["tint"] if mode == "Light" else None))
    radar.update_layout(title="Student performance profile",
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
    style(radar)

    bar = px.bar(
        pd.DataFrame({"Factor": ["Study", "Attendance", "Mental health", "Exercise"],
                      "Value": [study_hours_per_day * 8, attendance_percentage,
                                mental_health_rating * 10, exercise_frequency * 14]}),
        x="Factor", y="Value", color="Factor", title="Academic performance factors")
    bar.update_layout(showlegend=False)
    style(bar)

    tab1, tab2, tab3 = st.tabs(["Score & time", "Profile", "Factors"])
    with tab1:
        a, b = st.columns(2)
        a.plotly_chart(gauge, use_container_width=True)
        b.plotly_chart(pie, use_container_width=True)
    with tab2:
        st.plotly_chart(radar, use_container_width=True)
    with tab3:
        st.plotly_chart(bar, use_container_width=True)

    # ---- Recommendations ----
    tips = []
    if study_hours_per_day < 3:
        tips.append("📖 Increase study hours")
    if social_media_hours > 3:
        tips.append("📱 Reduce social media usage")
    if netflix_hours > 2.5:
        tips.append("🎬 Reduce Netflix time")
    if sleep_hours < 6:
        tips.append("😴 Sleep at least 7 hours")
    if sleep_hours >= 9:
        tips.append("😴 Avoid oversleeping")
    if attendance_percentage < 80:
        tips.append("🏫 Improve attendance")
    if exercise_frequency < 2:
        tips.append("🏃 Exercise more often")
    if mental_health_rating < 5:
        tips.append("🧠 Focus on mental health")

    if tips:
        st.subheader("💡 Recommendations")
        for tip in tips:
            st.markdown(f'<div class="tip">{tip}</div>', unsafe_allow_html=True)

    st.caption("Model prediction error ≈ ±5.5 marks")