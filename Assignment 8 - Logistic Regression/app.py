import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Prediction (Logistic Regression)", layout="centered")

st.title("Diabetes Prediction — Logistic Regression")
st.write("Enter patient details to predict the probability of diabetes (Outcome=1).")

@st.cache_resource
def load_artifacts():
    model = joblib.load("logreg_pipeline.joblib")
    features = json.load(open("features.json"))
    return model, features

model, features = load_artifacts()

st.subheader("Input Features")

# Default values are based on typical ranges; adjust if you want.
input_data = {}

# Create a simple UI: sliders for numeric features
for col in features:
    # Reasonable min/max based on known dataset ranges (safe defaults)
    if col == "Pregnancies":
        input_data[col] = st.number_input(col, min_value=0, max_value=20, value=1, step=1)
    elif col == "Glucose":
        input_data[col] = st.number_input(col, min_value=0, max_value=250, value=120, step=1)
    elif col == "BloodPressure":
        input_data[col] = st.number_input(col, min_value=0, max_value=200, value=70, step=1)
    elif col == "SkinThickness":
        input_data[col] = st.number_input(col, min_value=0, max_value=100, value=20, step=1)
    elif col == "Insulin":
        input_data[col] = st.number_input(col, min_value=0, max_value=900, value=80, step=1)
    elif col == "BMI":
        input_data[col] = st.number_input(col, min_value=0.0, max_value=70.0, value=28.0, step=0.1)
    elif col == "DiabetesPedigreeFunction":
        input_data[col] = st.number_input(col, min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    elif col == "Age":
        input_data[col] = st.number_input(col, min_value=1, max_value=120, value=33, step=1)
    else:
        # fallback
        input_data[col] = st.number_input(col, value=0.0)

input_df = pd.DataFrame([input_data], columns=features)

st.divider()

if st.button("Predict"):
    prob = float(model.predict_proba(input_df)[0][1])
    pred = int(prob >= 0.5)
    st.subheader("Prediction")
    st.write(f"**Predicted class:** {pred} (1 = Diabetes, 0 = No Diabetes)")
    st.write(f"**Predicted probability (Outcome=1):** {prob:.3f}")

    if pred == 1:
        st.warning("Model indicates higher risk of diabetes. This is a demo model; not medical advice.")
    else:
        st.success("Model indicates lower risk of diabetes. This is a demo model; not medical advice.")

st.caption("Note: This app is for learning/deployment practice only.")
