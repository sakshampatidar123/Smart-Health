import streamlit as st
import pickle
import numpy as np

# Load the trained Random Forest model
model_filename = 'rf_tuned.sav'
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Load the saved StandardScaler
scaler_filename = 'scaler.sav'
with open(scaler_filename, 'rb') as file:
    loaded_scaler = pickle.load(file)

# Set page configuration
st.set_page_config(page_title="Smart Health", page_icon="🩺", layout="wide")

# Sidebar Navigation
st.sidebar.title("Smart Health")
page = st.sidebar.radio("Select a Page", ["Home", "Diabetes Prediction System"])

if page == "Home":
    st.title("🏥 Welcome to Smart Health")
    st.write(
        "This AI-powered system helps predict diabetes based on user input. Select **Diabetes Prediction System** from the sidebar to begin.")
    st.image(
        "https://www.kindpng.com/picc/m/190-1905381_healthcare-clipart-transparent-background-medical-health-icon-hd.png",
        width=300)
    st.markdown("---")
    st.write("### 🔍 Features:")
    st.write("✅ User-friendly input interface")
    st.write("✅ AI-powered diabetes prediction")
    st.write("✅ Confidence Score for prediction")
    st.write("✅ Secure and reliable")

elif page == "Diabetes Prediction System":
    st.title("🩺 Diabetes Prediction System")
    st.write("Enter the details below to predict whether a person has diabetes.")

    # User Input Fields with default value 0
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=0)
    insulin = st.number_input("Insulin Level", min_value=0.0, max_value=900.0, value=0.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=0.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.0)
    age = st.number_input("Age", min_value=0, max_value=120, value=0)

    # Convert input into numpy array
    input_data_raw = np.array(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

    # Scale the input data
    input_data_scaled = loaded_scaler.transform(input_data_raw)

    # Prediction Button
    if st.button("Predict Diabetes"):
        try:
            prediction = loaded_model.predict(input_data_scaled)
            confidence_score = loaded_model.predict_proba(input_data_scaled)[:, 1]  # Probability of being diabetic

            # Show the result
            if prediction[0] == 1:
                st.error(f"🩺 The person is **Diabetic** (Confidence Score: {confidence_score[0]:.4f})")
            else:
                st.success(f"✅ The person is **Not Diabetic** (Confidence Score: {confidence_score[0]:.4f})")

        except Exception as e:
            st.error("❌ Error making prediction: " + str(e))

    # Footer
    st.markdown("---")
    st.write("🔬 Developed by [Saksham Patidar] | Powered by Machine Learning")
