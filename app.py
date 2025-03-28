# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd

# # Load the model and threshold
# model = joblib.load('stroke_prediction_model.pkl')
# with open('best_threshold.txt', 'r') as f:
#     best_threshold = float(f.read())

# # Define the risk scoring function
# def assign_risk_score(prob):
#     if prob < best_threshold:
#         return "Low Risk"
#     elif best_threshold <= prob <= 0.15:
#         return "Medium Risk"
#     else:
#         return "High Risk"

# # Streamlit app
# st.title("Stroke Risk Prediction App")
# st.write("Enter patient details to predict stroke risk.")

# # Input fields for features (adjust based on your dataset's features)
# age = st.number_input("Age", min_value=0, max_value=120, value=50)
# gender = st.selectbox("Gender", options=["Male", "Female"])
# hypertension = st.selectbox("Hypertension", options=["No", "Yes"])
# heart_disease = st.selectbox("Heart Disease", options=["No", "Yes"])
# ever_married = st.selectbox("Ever Married", options=["No", "Yes"])
# work_type = st.selectbox("Work Type", options=["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
# residence_type = st.selectbox("Residence Type", options=["Urban", "Rural"])
# avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
# bmi = st.number_input("BMI", min_value=0.0, value=25.0)
# smoking_score = st.selectbox("Smoking Score", options=[0, 1, 2])  # 0: never, 1: formerly, 2: smokes

# # Derived features (based on your feature engineering)
# age_glucose = age * avg_glucose_level
# comorbidity = 1 if hypertension == "Yes" or heart_disease == "Yes" else 0
# age_group = "Young" if age < 40 else "Middle" if age < 60 else "Old"

# # Map categorical variables to numerical (same as in your preprocessing)
# gender_map = {"Male": 1, "Female": 0}
# hypertension_map = {"No": 0, "Yes": 1}
# heart_disease_map = {"No": 0, "Yes": 1}
# ever_married_map = {"No": 0, "Yes": 1}
# work_type_map = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "Children": 3, "Never_worked": 4}
# residence_type_map = {"Urban": 1, "Rural": 0}
# age_group_map = {"Young": 0, "Middle": 1, "Old": 2}

# # Create input array for prediction
# input_data = np.array([[
#     age,
#     gender_map[gender],
#     hypertension_map[hypertension],
#     heart_disease_map[heart_disease],
#     ever_married_map[ever_married],
#     work_type_map[work_type],
#     residence_type_map[residence_type],
#     avg_glucose_level,
#     bmi,
#     smoking_score,
#     age_glucose,
#     comorbidity,
#     age_group_map[age_group]
# ]])

# # Predict
# if st.button("Predict"):
#     prob = model.predict_proba(input_data)[0, 1]  # Probability of stroke
#     risk_category = assign_risk_score(prob)
    
#     st.write(f"**Predicted Stroke Probability**: {prob:.4f}")
#     st.write(f"**Risk Category**: {risk_category}")
#     if risk_category == "High Risk":
#         st.warning("This patient is at high risk of stroke. Immediate action recommended.")
#     elif risk_category == "Medium Risk":
#         st.warning("This patient is at medium risk of stroke. Monitor closely.")
#     else:
#         st.success("This patient is at low risk of stroke.")
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# Load the model and threshold
model = joblib.load('stroke_prediction_model.pkl')
with open('best_threshold.txt', 'r') as f:
    best_threshold = float(f.read())

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

# Define the risk scoring function
def assign_risk_score(prob):
    if prob < best_threshold:
        return "Low Risk"
    elif best_threshold <= prob <= 0.15:
        return "Medium Risk"
    else:
        return "High Risk"

# Streamlit app
st.title("Stroke Risk Prediction App")
st.write("Enter patient details to predict stroke risk and understand contributing factors.")

# --- Model Performance Summary ---
st.subheader("Model Performance Summary")
st.write("This model was trained on an imbalanced dataset (4,861 non-stroke vs. 249 stroke cases).")
st.write(f"- **F1-Score**: 0.2952 ( limited by imbalance)")
st.write(f"- **Precision**: 0.1938 (many false positives, but acceptable in medical context)")
st.write(f"- **Recall**: 0.62 (catches 62% of stroke cases, a key strength)")
st.write(f"- **ROC-AUC**: 0.8113 (good discriminative power)")
# st.write("**Limitations**: Low precision due to imbalance; may overpredict stroke. Use predictions as a guide, not a definitive diagnosis.")

# --- Input Fields ---
st.subheader("Patient Details")
age = st.number_input("Age", min_value=0, max_value=120, value=50)
gender = st.selectbox("Gender", options=["Male", "Female"])
hypertension = st.selectbox("Hypertension", options=["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", options=["No", "Yes"])
ever_married = st.selectbox("Ever Married", options=["No", "Yes"])
work_type = st.selectbox("Work Type", options=["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
residence_type = st.selectbox("Residence Type", options=["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
# Change 1: Update smoking status to verbal options
smoking_status = st.selectbox("Smoking Status", options=["Never", "Formerly", "Smokes"])

# Derived features
age_glucose = age * avg_glucose_level
comorbidity = 1 if hypertension == "Yes" or heart_disease == "Yes" else 0
age_group = "Young" if age < 40 else "Middle" if age < 60 else "Old"

# Map categorical variables to numerical
gender_map = {"Male": 1, "Female": 0}
hypertension_map = {"No": 0, "Yes": 1}
heart_disease_map = {"No": 0, "Yes": 1}
ever_married_map = {"No": 0, "Yes": 1}
work_type_map = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "Children": 3, "Never_worked": 4}
residence_type_map = {"Urban": 1, "Rural": 0}
age_group_map = {"Young": 0, "Middle": 1, "Old": 2}
# Change 1: Map verbal smoking status to numerical values
smoking_map = {"Never": 0, "Formerly": 1, "Smokes": 2}
smoking_score = smoking_map[smoking_status]

# Create input array for prediction
input_data = np.array([[
    age,
    gender_map[gender],
    hypertension_map[hypertension],
    heart_disease_map[heart_disease],
    ever_married_map[ever_married],
    work_type_map[work_type],
    residence_type_map[residence_type],
    avg_glucose_level,
    bmi,
    smoking_score,
    age_glucose,
    comorbidity,
    age_group_map[age_group]
]])

# Feature names for SHAP (same order as input_data, excluding gender for display)
feature_names = [
    "age", 
    "gender",  # Included in input_data for prediction
    "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "avg_glucose_level", "bmi",
    "smoking_score", "age_glucose", "comorbidity", "age_group"
]

# Change 2: Feature names for SHAP display (excluding gender)
shap_feature_names = [
    "age",
    "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "avg_glucose_level", "bmi",
    "smoking_score", "age_glucose", "comorbidity", "age_group"
]

# --- Predict and Display Results ---
if st.button("Predict"):
    # Predict probability
    prob = model.predict_proba(input_data)[0, 1]
    risk_category = assign_risk_score(prob)

    # Display prediction
    st.subheader("Prediction Results")
    st.write(f"**Predicted Stroke Probability**: {prob:.4f}")
    st.write(f"**Risk Category**: {risk_category}")
    if risk_category == "High Risk":
        st.warning("This patient is at high risk of stroke. Immediate action recommended.")
    elif risk_category == "Medium Risk":
        st.warning("This patient is at medium risk of stroke. Monitor closely.")
    else:
        st.success("This patient is at low risk of stroke.")

    # --- SHAP Explanation ---
    st.subheader("Why This Prediction? (SHAP Explanation)")
    shap_values = explainer.shap_values(input_data)
    # Change 2: Exclude gender (index 1) from SHAP values and input data for display
    shap_values_display = np.delete(shap_values, 1, axis=1)  # Remove gender column from SHAP values
    input_data_display = np.delete(input_data, 1, axis=1)  # Remove gender column from input data
    # Create a bar plot for SHAP values
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values_display, input_data_display, feature_names=shap_feature_names, plot_type="bar", show=False)
    st.pyplot(fig)
    st.write("This plot shows the contribution of each feature to the prediction (excluding gender). Positive values increase stroke risk, negative values decrease it.")

    # --- Risk Score Trend Visualization ---
    st.subheader("Risk Trend Analysis")
    st.write("See how the risk changes with a key feature (e.g., Average Glucose Level).")
    glucose_range = np.linspace(50, 250, 50)  # Range of glucose levels to test
    probs = []
    for glucose in glucose_range:
        temp_input = input_data.copy()
        temp_input[0, 7] = glucose  # Update avg_glucose_level
        temp_input[0, 10] = age * glucose  # Update age_glucose
        prob_temp = model.predict_proba(temp_input)[0, 1]
        probs.append(prob_temp)

    # Plot the trend
    fig, ax = plt.subplots()
    ax.plot(glucose_range, probs, marker='o')
    ax.set_xlabel("Average Glucose Level")
    ax.set_ylabel("Stroke Probability")
    ax.set_title("Stroke Risk vs. Average Glucose Level")
    ax.grid(True)
    st.pyplot(fig)
    st.write("This plot shows how the stroke probability changes as the average glucose level varies, keeping other features constant.")

    # --- Export Results ---
    st.subheader("Export Results")
    result_dict = {
        "Age": age,
        "Gender": gender,
        "Hypertension": hypertension,
        "Heart Disease": heart_disease,
        "Ever Married": ever_married,
        "Work Type": work_type,
        "Residence Type": residence_type,
        "Average Glucose Level": avg_glucose_level,
        "BMI": bmi,
        "Smoking Status": smoking_status,  # Change 1: Use verbal smoking status in export
        "Probability": prob,
        "Risk Category": risk_category
    }
    result_df = pd.DataFrame([result_dict])
    csv = result_df.to_csv(index=False)
    st.download_button(
        label="Download Prediction Results as CSV",
        data=csv,
        file_name="stroke_prediction_result.csv",
        mime="text/csv"
    )