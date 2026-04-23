# =========================
# STREAMLIT DEPLOYMENT APP
# Depression Prediction System
# =========================

import streamlit as st
import numpy as np
import joblib
import pandas as pd

# =========================
# 1. LOAD MODEL + ENCODER
# =========================
model = joblib.load("depression_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# =========================
# 2. PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Depression Prediction AI",
    layout="centered"
)

st.title("🧠 Depression Prediction System")
st.write("Enter symptom levels below to predict mental health state.")

# =========================
# 3. FEATURE INPUTS
# =========================
st.header("Patient Symptom Inputs")

sleep = st.slider("Sleep", 0.0, 10.0, 5.0)
appetite = st.slider("Appetite", 0.0, 10.0, 5.0)
interest = st.slider("Interest", 0.0, 10.0, 5.0)
fatigue = st.slider("Fatigue", 0.0, 10.0, 5.0)
worthlessness = st.slider("Worthlessness", 0.0, 10.0, 5.0)
concentration = st.slider("Concentration", 0.0, 10.0, 5.0)
agitation = st.slider("Agitation", 0.0, 10.0, 5.0)
suicidal_ideation = st.slider("Suicidal Ideation", 0.0, 10.0, 0.0)
sleep_disturbance = st.slider("Sleep Disturbance", 0.0, 10.0, 5.0)
aggression = st.slider("Aggression", 0.0, 10.0, 3.0)
panic_attacks = st.slider("Panic Attacks", 0.0, 10.0, 2.0)
hopelessness = st.slider("Hopelessness", 0.0, 10.0, 5.0)
restlessness = st.slider("Restlessness", 0.0, 10.0, 5.0)
low_energy = st.slider("Low Energy", 0.0, 10.0, 5.0)

# =========================
# 4. INPUT VECTOR
# =========================
input_data = np.array([[
    sleep, appetite, interest, fatigue,
    worthlessness, concentration, agitation,
    suicidal_ideation, sleep_disturbance,
    aggression, panic_attacks, hopelessness,
    restlessness, low_energy
]])

# =========================
# 5. PREDICTION
# =========================
if st.button("Predict Depression State"):

    prediction = model.predict(input_data)
    prediction_label = label_encoder.inverse_transform(prediction)[0]

    st.subheader("🧾 Prediction Result")

    if prediction_label.lower() == "no depression":
        st.success(f"Result: {prediction_label}")
    elif prediction_label.lower() == "mild":
        st.info(f"Result: {prediction_label}")
    elif prediction_label.lower() == "moderate":
        st.warning(f"Result: {prediction_label}")
    else:
        st.error(f"Result: {prediction_label}")

    # =========================
    # 6. PROBABILITY DISPLAY
    # =========================
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]

        prob_df = pd.DataFrame({
            "Class": label_encoder.classes_,
            "Probability": probs
        })

        st.subheader("📊 Prediction Probabilities")
        st.bar_chart(prob_df.set_index("Class"))
# streamlit run app.py