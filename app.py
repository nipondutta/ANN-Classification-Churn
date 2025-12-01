import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle

# Loading the trained models
model = tf.keras.models.load_model('model.h5')
onehot_encoder_geo = pickle.load(open('onehot_encoder_geo.pkl', 'rb'))
label_encoder_gender = pickle.load(open('label_encoder_gender.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Streamlit App
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox("Select Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Select Gender", label_encoder_gender.classes_)
age = st.slider("Select Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Enter Balance", min_value=0.0, format="%.2f")
credit_score = st.number_input("Enter Credit Score", min_value=300, max_value=900, value=600)
tenure = st.slider("Select Tenure", min_value=0, max_value=10, value=3)
num_of_products = st.slider("Select Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card?", options=[0, 1])
is_active_member = st.selectbox("Is Active Member?", options=[0, 1])
estimated_salary = st.number_input("Enter Estimated Salary", min_value=0.0, format="%.2f")

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Onehot encode geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded, 
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Concatenate all features
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)
pred_probability = prediction[0][0]

# Output result
if pred_probability > 0.5:
    st.success(f"🚨 Customer likely to **leave** the bank!\nChurn Probability: {pred_probability:.2f}")
else:
    st.info(f"😊 Customer likely to **stay** with the bank.\nProbability: {(1 - pred_probability):.2f}")
