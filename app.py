import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ---- App Title ----
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💼 Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ---- Input Section ----
st.markdown("### 🧾 Enter Customer Information Below")

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('🌍 Select Geography', onehot_encoder_geo.categories_[0])
    credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=900, step=1)
    balance = st.number_input('🏦 Account Balance')
    num_of_products = st.slider('📦 Number of Products', 1, 4)
    is_active_member = st.selectbox('🟢 Is Active Member?', [0, 1])

with col2:
    gender = st.selectbox('⚥ Select Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92)
    tenure = st.slider('📆 Tenure (in Years)', 0, 10)
    estimated_salary = st.number_input('💰 Estimated Salary')
    has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1])

# ---- Prepare Data ----
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

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data)

# ---- Prediction ----
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 🔍 Prediction Result")

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.markdown(
    f"<div style='text-align: center; font-size: 20px; color: #1E88E5;'>"
    f"📊 <strong>Churn Probability: {prediction_proba:.2f}</strong>"
    f"</div>",
    unsafe_allow_html=True
)

if prediction_proba > 0.5:
    st.markdown("<div style='text-align: center; color: red; font-size: 18px;'>⚠️ The customer is <strong>likely</strong> to leave the bank.</div>", unsafe_allow_html=True)
else:
    st.markdown("<div style='text-align: center; color: green; font-size: 18px;'>✅ The customer is <strong>not likely</strong> to leave the bank.</div>", unsafe_allow_html=True)
