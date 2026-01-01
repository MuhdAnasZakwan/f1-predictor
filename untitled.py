import streamlit as st
import joblib
import numpy as np

model = joblib.load('top3_rf_model.joblib')

st.title('Formula 1 Podium Prediction App')

