import streamlit as st
import numpy as np
import joblib

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS to style the app
# 2. Custom CSS to style the app
st.markdown("""
    <style>
    /* Force the main background to a light color to match the "clean" look */
    .main {
        background-color: #f8f9fa; 
    }
    
    /* Style the buttons */
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }

    /* FIX: Style the Metric Card */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }

    /* FIX: Force text inside the metric to be black */
    div[data-testid="stMetric"] > label {
        color: #333333 !important; /* Label color */
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #000000 !important; /* Value color */
    }
    </style>
    """, unsafe_allow_html=True)

# Load resources (Wrapped in a function to allow caching in the future)
@st.cache_resource
def load_resources():
    try:
        model = joblib.load("logistic_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model files (logistic_model.pkl, scaler.pkl) not found.")
        return None, None

model, scaler = load_resources()

# --- Header Section ---
st.title("👥 Employee Attrition Prediction")
st.markdown("### HR Management Dashboard")
st.markdown("---")

if model and scaler:
    # --- Input Section (Organized into 3 Columns) ---
    st.subheader("📝 Employee Details")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("Personal & History")
        age = st.number_input("Age", 18, 60, 34)
        distance_from_home = st.number_input("Distance from home (km)", 1, 25, 1)
        num_companies_worked = st.number_input("Num. companies worked", 0, 9, 1)
        total_working_years = st.number_input("Total working years", 0, 40, 10)

    with col2:
        st.info("Role & Satisfaction")
        years_at_company = st.number_input("Years at company", 0, 40, 5)
        years_in_current_role = st.number_input("Years in current role", 0, 15, 2)
        environment_satisfaction = st.slider("Environment Satisfaction (1-Low to 4-High)", 1, 4, 4)
        overtime = st.selectbox("Overtime", ['No', 'Yes']) # Swapped order for UI, logic handled below

    with col3:
        st.info("Compensation & Rates")
        monthly_rate = st.number_input("Monthly Rate", 2100, 26500,value=4223)
        daily_rate = st.number_input("Daily Rate", 110, 1400, value=700)
        hourly_rate = st.number_input("Hourly Rate", 30, 100, value=48)
        percent_salary_hike = st.number_input("Percent salary hike (%)", 1, 13, 12)

    st.markdown("---")

    # --- Prediction Section ---
    # Center the button using columns
    _, btn_col, _ = st.columns([1, 2, 1])
    
    with btn_col:
        predict_btn = st.button("🔍 Analyze Attrition Risk")

    # --- Core Logic (Preserved) ---
    # converting overtime (Logic kept exactly as requested)
    overtime_val = 1 if overtime == 'Yes' else 0

    if predict_btn:
        # EXACT input structure required by your model
        input_data = np.array([[
            age, total_working_years, overtime_val, daily_rate, hourly_rate,
            monthly_rate, distance_from_home, years_at_company, percent_salary_hike,
            years_in_current_role, environment_satisfaction, num_companies_worked
        ]])

        input_scaled = scaler.transform(input_data)
        
        # predict probability
        prediction_proba = model.predict_proba(input_scaled)[0][1]  # first row, 2nd column
        threshold = 0.5

        # --- Enhanced Output UI ---
        st.markdown("### 📊 Prediction Results")
        
        # visual gauge for probability
        res_col1, res_col2 = st.columns([1, 3])
        
        with res_col1:
            st.metric(label="Attrition Probability", value=f"{prediction_proba*100:.1f}%")
        
        with res_col2:
            st.write("Risk Level:")
            # Dynamic progress bar color logic
            if prediction_proba > threshold:
                bar_color = "red"
                st.progress(float(prediction_proba), text="High Risk")
                st.error("⚠ **Alert: Employee is Likely to Quit**")
            else:
                bar_color = "green"
                st.progress(float(prediction_proba), text="Low Risk")
                st.success("✅ **Safe: Employee is Likely to Stay**")
