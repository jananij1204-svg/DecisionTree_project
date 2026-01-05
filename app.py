import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------------------------
# Load Model
# ---------------------------
MODEL_PATH = "decision.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file 'decision.pkl' not found. Place it next to app.py")
    st.stop()

st.title("🎬 Movie Interest Prediction App")
st.write("Enter the details below to predict movie interest")

# ---------------------------
# Detect Features Automatically
# ---------------------------

if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
else:
    st.error("❌ Model does not contain feature names. Please retrain with feature names.")
    st.stop()

st.info(f"Detected Features: {', '.join(feature_names)}")

# ---------------------------
# Auto-Generate Input Fields
# ---------------------------

user_inputs = {}

for feature in feature_names:

    # If feature is numeric
    if "age" in feature.lower():
        user_inputs[feature] = st.slider(feature, 0, 100, 25)

    elif "rating" in feature.lower():
        user_inputs[feature] = st.slider(feature, 0, 10, 5)

    elif "interest" in feature.lower():
        user_inputs[feature] = st.slider(feature, 0, 5, 2)

    # For categorical strings
    elif feature.lower() in ["gender", "sex"]:
        user_inputs[feature] = st.selectbox(feature, ["male", "female"])

    elif feature.lower() in ["genre", "movie_genre"]:
        user_inputs[feature] = st.selectbox(feature, 
                                            ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi"])

    else:
        # Default numeric input
        user_inputs[feature] = st.number_input(feature, value=0.0)


# ---------------------------
# Convert categorical values
# ---------------------------

row = {}

for feature, value in user_inputs.items():
    if isinstance(value, str):
        # Convert categorical to integer labels
        row[feature] = pd.factorize([value])[0][0]
    else:
        row[feature] = value

input_df = pd.DataFrame([row])

# ---------------------------
# Prediction
# ---------------------------

if st.button("Predict Movie Interest"):
    try:
        pred = model.predict(input_df)[0]

        try:
            prob = model.predict_proba(input_df)[0]
        except:
            prob = None

        st.success(f"🎯 Prediction: {pred}")

        if prob is not None:
            st.info(f"📊 Probability: {prob}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
