import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# Define the Hugging Face repo details for the model
model_repo_id = "satishp0879/tourism-customer-model"
model_filename = "Tourism_customer_targetting_best_model_V1.joblib"
model_repo_type = "model" # The model was uploaded to a 'model' type repository

# Download and load the model
try:
    # Ensure HF_TOKEN is available if required by hf_hub_download
    # In HF Spaces, HF_TOKEN might be automatically handled if set as a Space Secret.
    model_path = hf_hub_download(repo_id=model_repo_id, filename=model_filename, repo_type=model_repo_type)
    model = joblib.load(model_path)
    st.success("Model loaded successfully from Hugging Face!")
except Exception as e:
    st.error(f"Failed to load model from Hugging Face: {e}")
    st.warning("Please ensure the model file exists in the Hugging Face repository and HF_TOKEN (if required) is correctly set.")
    st.stop() # Stop the app if model loading fails

# --- Streamlit UI for Tourism Customer Targetting ----
st.title("Tourism Customer Targetting App")
st.write("""
This application uses a machine learning model to predict whether a customer will purchase the newly introduced Wellness Tourism Package based on their information.
""")

# Define numeric and categorical features (must match prep.py and train.py)
numeric_features = [
    'Age',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'PitchSatisfactionScore',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]
categorical_features = [
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Passport',
    'OwnCar',
    'Designation'
]

# --- Input Collection ---
st.header("Enter Customer Information")

customer_id_input = st.text_input("CustomerID", help="Must be 6 digits, numbers only, starting with '2'")
valid_customer_id = False
if customer_id_input:
    if len(customer_id_input) == 6 and customer_id_input.isdigit() and customer_id_input.startswith('2'):
        valid_customer_id = True
    else:
        st.error("CustomerID must be 6 digits long, contain only numbers, and start with '2'.")
else:
    st.warning("Please enter a CustomerID.")

inputs = {}

# Numeric inputs
st.subheader("Numeric Features")
inputs['Age'] = st.number_input("Age", min_value=18, max_value=90, value=30)
inputs['DurationOfPitch'] = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=10)
inputs['NumberOfPersonVisiting'] = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=2)
inputs['PreferredPropertyStar'] = st.number_input("Preferred Property Star (1-5)", min_value=1, max_value=5, value=3)
inputs['NumberOfTrips'] = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5)
inputs['PitchSatisfactionScore'] = st.number_input("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
inputs['NumberOfChildrenVisiting'] = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
inputs['MonthlyIncome'] = st.number_input("Monthly Income", min_value=0, max_value=500000, value=50000, step=1000)

# Categorical inputs
st.subheader("Categorical Features")
inputs['TypeofContact'] = st.selectbox("Type of Contact", ['Company Invited', 'Self Inquiry'])
inputs['CityTier'] = st.selectbox("City Tier", ['1', '2', '3']) # Assuming these are string representations
inputs['Occupation'] = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer', 'Unemployed'])
inputs['Gender'] = st.selectbox("Gender", ['Male', 'Female', 'Fe Male']) # Including 'Fe Male' as it can appear in datasets
inputs['ProductPitched'] = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'Luxury'])
inputs['MaritalStatus'] = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
inputs['Passport'] = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
inputs['OwnCar'] = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
inputs['Designation'] = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP', 'Director', 'CEO', 'Consultant', 'Team Leader']) # A more exhaustive list of examples

# --- Preprocessing Function (must mimic train.py's preprocess_data) ---
def preprocess_input_data(input_dict, categorical_features, model_feature_columns):
    # Create a DataFrame from the single input dictionary
    df = pd.DataFrame([input_dict])

    # Ensure all categorical columns are treated as objects before one-hot encoding
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Apply one-hot encoding with drop_first=True, consistent with training
    df_processed = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # Align columns with the model's expected features
    # Create a DataFrame with all expected columns, initialized to 0
    aligned_df = pd.DataFrame(0, index=df_processed.index, columns=model_feature_columns)

    # Fill in the values from the processed input
    for col in df_processed.columns:
        if col in aligned_df.columns:
            aligned_df[col] = df_processed[col]

    return aligned_df

# IMPORTANT: This list of feature names and their order MUST EXACTLY match the features
# and their order that the model was trained on. In a robust MLOps pipeline,
# these names would be saved as an artifact during training and loaded here.
# For demonstration, this list is constructed based on common `get_dummies` behavior.
# If your model fails to predict, this is the first place to check for discrepancies!
model_feature_columns = (
    numeric_features +
    ['TypeofContact_Self Inquiry',
     'CityTier_2', 'CityTier_3',
     'Occupation_Large Business', 'Occupation_Salaried', 'Occupation_Small Business', 'Occupation_Unemployed',
     'Gender_Male', 'Gender_Fe Male', # Assuming 'Female' was dropped alphabetically
     'ProductPitched_Deluxe', 'ProductPitched_Luxury', 'ProductPitched_Standard', 'ProductPitched_Super Deluxe',
     'MaritalStatus_Married', 'MaritalStatus_Single',
     'Passport_1',
     'OwnCar_1',
     # Remaining 4 columns for 'Designation' to match 29 total features (8 numeric + 21 categorical dummies)
     'Designation_Executive', 'Designation_Manager', 'Designation_Senior Manager', 'Designation_VP'
    ]
)

# --- Prediction Logic ---
if st.button("Predict Purchase"):
    if not valid_customer_id:
        st.error("Please correct the CustomerID before predicting.")
    else:
        try:
            # Preprocess the collected inputs
            processed_input = preprocess_input_data(inputs, categorical_features, model_feature_columns)

            # Make prediction
            prediction = model.predict(processed_input)[0]
            prediction_proba = model.predict_proba(processed_input)[0]

            st.subheader("Prediction Result:")
            if prediction == 1:
                st.success(f"The model predicts: **Customer WILL purchase the Wellness Tourism Package!**")
            else:
                st.info(f"The model predicts: **Customer will NOT purchase the Wellness Tourism Package.**")

            st.write(f"Probability of Purchase: {prediction_proba[1]:.2f}")
            st.write(f"Probability of No Purchase: {prediction_proba[0]:.2f}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please ensure all inputs are valid and the model's expected features are correctly aligned.")

st.markdown("---")
st.markdown("Developed by MLOps Engineer")
