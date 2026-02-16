import streamlit as st
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Attrition Predictor",
    page_icon="📊",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root variables ── */
:root {
    --navy:       #0b0f1a;
    --navy-card:  #111827;
    --navy-input: #1a2235;
    --border:     #1e2d45;
    --amber:      #f59e0b;
    --amber-glow: rgba(245,158,11,0.15);
    --amber-dark: #d97706;
    --text:       #e2e8f0;
    --muted:      #64748b;
    --success:    #10b981;
    --danger:     #ef4444;
}

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--navy) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2.5rem 1.5rem 4rem !important;
    max-width: 760px !important;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 2rem;
    margin-bottom: 2rem;
}
.hero-tag {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--amber);
    background: var(--amber-glow);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 2px;
    padding: 0.3rem 0.75rem;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    line-height: 1.15;
    color: #f1f5f9;
    margin: 0 0 0.6rem;
    letter-spacing: -0.02em;
}
.hero h1 span { color: var(--amber); }
.hero p {
    font-size: 0.95rem;
    color: var(--muted);
    margin: 0;
    font-weight: 400;
}

/* ── Section label ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--amber);
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Card wrapper ── */
.card {
    background: var(--navy-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
}

/* ── Streamlit widget overrides ── */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div,
[data-testid="stSlider"] {
    background: var(--navy-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 3px var(--amber-glow) !important;
}

/* Widget labels */
label, [data-testid="stWidgetLabel"] p {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
}

/* Slider track & thumb */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--amber) !important;
}
[data-testid="stSlider"] > div > div > div {
    background: var(--navy-input) !important;
}

/* ── Predict button ── */
[data-testid="stButton"] > button {
    width: 100%;
    padding: 0.85rem 2rem;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--navy) !important;
    background: var(--amber) !important;
    border: none !important;
    border-radius: 6px !important;
    cursor: pointer;
    transition: background 0.2s, transform 0.1s, box-shadow 0.2s;
    box-shadow: 0 4px 24px rgba(245,158,11,0.25);
    margin-top: 1.25rem;
}
[data-testid="stButton"] > button:hover {
    background: var(--amber-dark) !important;
    box-shadow: 0 6px 32px rgba(245,158,11,0.4);
    transform: translateY(-1px);
}
[data-testid="stButton"] > button:active { transform: translateY(0); }

/* ── Result card ── */
.result-box {
    border-radius: 10px;
    padding: 1.75rem 2rem;
    margin-top: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-box.stay {
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.35);
}
.result-box.quit {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.35);
}
.result-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.result-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.5rem;
    margin: 0 0 0.4rem;
}
.result-title.stay { color: #34d399; }
.result-title.quit { color: #f87171; }
.result-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
}

/* ── Probability bar ── */
.prob-bar-wrap {
    margin-top: 1.25rem;
    background: var(--navy-input);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
}
.prob-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.6s ease;
}
.prob-bar-fill.stay { background: linear-gradient(90deg, #059669, #34d399); }
.prob-bar-fill.quit { background: linear-gradient(90deg, #dc2626, #f87171); }
.prob-label {
    display: flex;
    justify-content: space-between;
    margin-top: 0.4rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
}

/* ── Divider ── */
.hr { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }

/* ── Tooltip-like helper text ── */
.helper {
    font-size: 0.7rem;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    margin-top: -0.75rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("logistic_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">HR Analytics · ML Model</div>
    <h1>Employee <span>Attrition</span><br>Predictor</h1>
    <p>Enter employee details to estimate flight risk using a trained logistic regression model.</p>
</div>
""", unsafe_allow_html=True)

# ── Section 1: Demographics ───────────────────────────────────────────────────
st.markdown('<div class="section-label">01 — Demographics</div>', unsafe_allow_html=True)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=60, value=30)
    with col2:
        distance_from_home = st.number_input("Distance from Home (km)", min_value=0, max_value=50, value=10)

    col3, col4 = st.columns(2)
    with col3:
        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=5)
    with col4:
        num_companies_worked = st.number_input("No. of Companies Worked", min_value=0, max_value=20, value=1)

# ── Section 2: Compensation ───────────────────────────────────────────────────
st.markdown('<div class="section-label">02 — Compensation</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    daily_rate = st.number_input("Daily Rate ($)", min_value=0, value=500)
with col6:
    hourly_rate = st.number_input("Hourly Rate ($)", min_value=0, value=50)

col7, col8 = st.columns(2)
with col7:
    monthly_rate = st.number_input("Monthly Rate ($)", min_value=0, value=10000)
with col8:
    percent_salary_hike = st.number_input("Salary Hike (%)", min_value=0, max_value=50, value=15)

# ── Section 3: Role & Satisfaction ───────────────────────────────────────────
st.markdown('<div class="section-label">03 — Role & Satisfaction</div>', unsafe_allow_html=True)
col9, col10 = st.columns(2)
with col9:
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=3)
with col10:
    years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=30, value=2)

col11, col12 = st.columns(2)
with col11:
    overtime = st.selectbox("Overtime", ["No", "Yes"])
with col12:
    environment_satisfaction = st.slider("Environment Satisfaction", min_value=1, max_value=4, value=3,
                                          help="1 = Low · 2 = Medium · 3 = High · 4 = Very High")

st.markdown('<p class="helper">1 = Low &nbsp;·&nbsp; 2 = Medium &nbsp;·&nbsp; 3 = High &nbsp;·&nbsp; 4 = Very High</p>', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
predict_clicked = st.button("⚡ Run Prediction")

if predict_clicked:
    overtime_val = 1 if overtime == "Yes" else 0

    input_data = np.array([[
        age, total_working_years, overtime_val, daily_rate, hourly_rate,
        monthly_rate, distance_from_home, years_at_company, percent_salary_hike,
        years_in_current_role, environment_satisfaction, num_companies_worked
    ]])

    input_scaled        = scaler.transform(input_data)
    prediction_proba    = model.predict_proba(input_scaled)[0][1]
    pct                 = round(prediction_proba * 100, 1)
    will_quit           = prediction_proba > 0.5
    cls                 = "quit" if will_quit else "stay"
    icon                = "⚠️" if will_quit else "✅"
    verdict             = "Likely to Quit" if will_quit else "Likely to Stay"

    st.markdown(f"""
    <div class="result-box {cls}">
        <div class="result-icon">{icon}</div>
        <div class="result-title {cls}">{verdict}</div>
        <div class="result-sub">Attrition probability: <strong>{pct}%</strong></div>
        <div class="prob-bar-wrap" style="margin-top:1rem;">
            <div class="prob-bar-fill {cls}" style="width:{pct}%"></div>
        </div>
        <div class="prob-label">
            <span>0%</span>
            <span>Threshold: 50%</span>
            <span>100%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)