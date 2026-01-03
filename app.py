import streamlit as st
import joblib
import pandas as pd

model = joblib.load('top3_rf_model.joblib')
features = joblib.load('model_features.joblib')

st.title('Formula 1 Podium Prediction App')

grid = st.number_input('Starting Grid Position', min_value=1, max_value=20, value=5)
year = st.number_input('Season Year', min_value=1950, max_value=2025, value=2023)
driver_podium_rate = st.slider('Driver Historical Podium Rate', min_value=0.0, max_value=1.0, value=0.3)
constructor_podium_rate = st.slider('Constructor Historical Podium Rate', min_value=0.0, max_value=1.0, value=0.4)

input_data = pd.DataFrame([{
    'grid':grid,
    'year':year,
    'driver_podium_rate':driver_podium_rate,
    'constructor_podium_rate':constructor_podium_rate
}])
input_data = input_data[features]

if st.button('Predict Podium Finish'):
    prob = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    st.write(f"Podium Probability: `{prob:.2%}`")
    if prediction == 1:
        st.success("Predicted: **Podium Finish (Top 3)**")
    else:
        st.warning("Predicted: **No Podium Finish**")