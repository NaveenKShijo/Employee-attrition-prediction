import streamlit as st
import numpy as np
import joblib

model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Employee Attrition Prediction")
st.write("Enter employee details below:")

# inputs

age = st.number_input("Age", 18, 60, 30)
total_working_years = st.number_input("Total working years", 0, 40, 5)
overtime = st.selectbox("Overtime", ['Yes', 'No'])
daily_rate = st.number_input("Daily Rate", value = 500)
hourly_rate = st.number_input("Hourly Rate", value = 50)
monthly_rate = st.number_input("Monthly Rate", value = 100000)
distance_from_home = st.number_input("Distance from home", 0, 50, 10)
years_at_company = st.number_input("Years at company", 0, 40, 3)
percent_salary_hike = st.number_input("Percent salary hike", 0, 50, 15)
years_in_current_role = st.number_input("Years in current role", 0, 30, 2)
environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1,4,3)
num_companies_worked = st.number_input("Number of companies worked", 0, 20, 1)

# converting overtime
overtime = 1 if overtime == 'Yes' else 0

if st.button("Predict"):
    input_data = np.array([[
        age, total_working_years, overtime, daily_rate, hourly_rate,
        monthly_rate, distance_from_home, years_at_company, percent_salary_hike,
        years_in_current_role, environment_satisfaction, num_companies_worked
    ]])

    input_scaled = scaler.transform(input_data)
    
    # predict probability
    prediction_proba = model.predict_proba(input_scaled)[0][1]  # first row, 2nd column
    threshold = 0.5

    if prediction_proba > threshold:
        st.error("⚠ Likely to Quit")
    else:
        st.success("✅ Likely to Stay")

    st.write(f"Attrition probability: {prediction_proba:.2f}")