import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Set the page title and description
st.title("Stroke Risk Prediction App (Simplified)")
st.write("""
This app predicts the risk of stroke for a patient based on their health data using a Logistic Regression model with SMOTE. 
Enter the patient details below and click 'Predict' to see the results, along with a SHAP explanation of the prediction.
""")

# Load the model and scaler
model = joblib.load('models1/logistic_regression_smote_model.pkl')
scaler = joblib.load('scalers/Logistic_scaler_smote_new.pkl')

# Create input fields for the user
st.header("Enter Patient Details")

# Numerical inputs
age = st.slider("Age (years)", min_value=0, max_value=120, value=63)
avg_glucose_level = st.slider("Average Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=78.23, step=0.01)
bmi = st.slider("BMI", min_value=10.0, max_value=50.0, value=34.8, step=0.1)

# Categorical inputs (dropdowns)
gender = st.selectbox("Gender", options=["Female", "Male", "Other"], index=1)
hypertension = st.selectbox("Hypertension", options=["No", "Yes"], index=0)
heart_disease = st.selectbox("Heart Disease", options=["No", "Yes"], index=0)
ever_married = st.selectbox("Ever Married", options=["No", "Yes"], index=1)
work_type = st.selectbox("Work Type", options=["Children", "Govt_job", "Never_worked", "Private", "Self-employed"], index=2)
Residence_type = st.selectbox("Residence Type", options=["Rural", "Urban"], index=0)
smoking_status = st.selectbox("Smoking Status", options=["Never smoked", "Formerly smoked", "Smokes", "Unknown"], index=2)

# Encode the categorical inputs
gender_map = {"Female": 0, "Male": 1, "Other": 2}
hypertension_map = {"No": 0, "Yes": 1}
heart_disease_map = {"No": 0, "Yes": 1}
ever_married_map = {"No": 0, "Yes": 1}
work_type_map = {"Children": 0, "Govt_job": 1, "Never_worked": 2, "Private": 3, "Self-employed": 4}
Residence_type_map = {"Rural": 0, "Urban": 1}
smoking_status_map = {"Never smoked": 0, "Formerly smoked": 1, "Smokes": 2, "Unknown": 3}

gender_encoded = gender_map[gender]
hypertension_encoded = hypertension_map[hypertension]
heart_disease_encoded = heart_disease_map[heart_disease]
ever_married_encoded = ever_married_map[ever_married]
work_type_encoded = work_type_map[work_type]
Residence_type_encoded = Residence_type_map[Residence_type]
smoking_status_encoded = smoking_status_map[smoking_status]

# Compute derived features
age_glucose = age * avg_glucose_level
comorbidity = 1 if hypertension_encoded == 1 or heart_disease_encoded == 1 else 0
age_group = 0 if age <= 40 else 1 if age <= 60 else 2

# Create a DataFrame with the input data
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender_encoded],
    'hypertension': [hypertension_encoded],
    'heart_disease': [heart_disease_encoded],
    'ever_married': [ever_married_encoded],
    'work_type': [work_type_encoded],
    'Residence_type': [Residence_type_encoded],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [smoking_status_encoded],
    'age_glucose': [age_glucose],
    'comorbidity': [comorbidity],
    'age_group': [age_group]
})

# Scale the input data
input_scaled = scaler.transform(input_data)
input_scaled_df = pd.DataFrame(input_scaled, columns=input_data.columns)

# Convert to NumPy array for prediction and SHAP (to avoid feature names warning)
input_scaled_array = input_scaled_df.to_numpy()

# Predict button
if st.button("Predict"):
    # Display the raw input values to confirm they are captured correctly
    st.header("Input Values (Raw)")
    st.write("The following raw input values were captured:")
    st.write(input_data)

    # Display the scaled input values
    st.header("Input Values (Scaled)")
    st.write("The following scaled input values were used for prediction:")
    st.write(input_scaled_df)

    # Make the prediction
    prediction = model.predict(input_scaled_array)
    prediction_proba = model.predict_proba(input_scaled_array)[0][1]  # Probability of stroke (class 1)

    # Display the prediction
    st.header("Prediction Results")
    if prediction[0] == 1:
        st.error(f"Stroke Risk: High ({prediction_proba*100:.2f}% probability)")
    else:
        st.success(f"Stroke Risk: Low ({prediction_proba*100:.2f}% probability)")

    # Generate SHAP explanation
    explainer = shap.LinearExplainer(model, input_scaled_array)
    shap_values = explainer.shap_values(input_scaled_array)

    # Display SHAP values numerically
    st.header("SHAP Values (Numerical)")
    st.write("The following SHAP values were computed for the prediction:")
    for feature, shap_val in zip(input_data.columns, shap_values[0]):
        st.write(f"{feature}: {shap_val:.6f}")

    # Display SHAP explanation (custom bar plot for a single sample)
    st.header("Explanation of Prediction")
    st.write("The following plot shows the factors contributing to the prediction. Positive values increase the risk of stroke, while negative values decrease the risk.")
    
    try:
        # Create a custom bar plot for the single sample
        plt.figure(figsize=(8, 4), dpi=80)
        features = input_data.columns
        values = shap_values[0]
        
        # Create bars
        colors = ['red' if val > 0 else 'blue' for val in values]
        plt.barh(features, values, color=colors)
        
        # Add labels, title, and grid
        plt.xlabel("SHAP Value (Impact on Prediction)")
        plt.title("SHAP Values for Stroke Prediction")
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Adjust x-axis limits to make bars visible
        max_abs_value = max(abs(min(values)), abs(max(values)))
        plt.xlim(-max_abs_value * 1.1, max_abs_value * 1.1)
        
        # Display the plot in Streamlit
        st.pyplot(plt.gcf())
        plt.close()  # Clear the plot to free memory
    except Exception as e:
        st.error(f"Error generating SHAP plot: {str(e)}")
        st.write("Displaying text-based explanation instead:")
        for feature, shap_val in zip(input_data.columns, shap_values[0]):
            st.write(f"{feature}: {shap_val:.6f}")

# Add a footer
st.write("---")
st.write("Built with Streamlit. For educational purposes only. Consult a healthcare professional for medical advice.")