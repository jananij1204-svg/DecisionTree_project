import streamlit as st
import pickle
import pandas as pd

# -------------------------
# Load model
# -------------------------
MODEL_PATH = "decision.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file 'decision.pkl' not found. Place it in the same folder as app.py")
    st.stop()

st.title("🚢 Titanic Survival Prediction - Decision Tree Model")
st.write("Enter passenger details to predict survival")

# --------------------------------
# USER INPUT FIELDS
# --------------------------------

passenger_id = st.number_input("Passenger ID", min_value=1, value=100)
p_class = st.selectbox("Passenger Class (p_class)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 30)
sib_sp = st.number_input("Siblings/Spouses Aboard (sib_sp)", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard (parch)", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# --------------------------------
# ENCODING (MUST MATCH TRAINING)
# --------------------------------

sex_map = {"male": 1, "female": 0}
embarked_map = {"S": 0, "C": 1, "Q": 2}

sex_encoded = sex_map[sex]
embarked_encoded = embarked_map[embarked]

# --------------------------------
# BUILD INPUT DATAFRAME
# --------------------------------

input_data = pd.DataFrame({
    "passenger_id": [passenger_id],
    "p_class": [p_class],
    "sex": [sex_encoded],
    "age": [age],
    "sib_sp": [sib_sp],
    "parch": [parch],
    "fare": [fare],
    "embarked": [embarked_encoded]
})

# --------------------------------
# PREDICTION
# --------------------------------

if st.button("Predict"):
    try:
        pred = model.predict(input_data)[0]

        if pred == 1:
            st.success("✔ The passenger SURVIVED")
        else:
            st.error("❌ The passenger DID NOT SURVIVE")

        # Probability (if available)
        try:
            prob = model.predict_proba(input_data)[0][1]
            st.info(f"Survival Probability: {prob:.2f}")
        except:
            pass

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
