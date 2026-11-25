import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle

model=tf.keras.models.load_model('model.h5')

##load the saved label encoder, onehot encoder and scaler
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('onehotencoder_geo.pkl', 'rb') as f:
    onehotencoder_geo = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


##streamlit app
st.title("Customer Churn Prediction")

#user input
geography = st.selectbox("Geography", onehotencoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.number_input("Age", 18,92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.number_input("Tenure",0,10)
num_of_products = st.number_input("Number of Products", 1,4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

#prepare the input data for prediction
input_data = pd.DataFrame({
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Balance': [balance],
    'CreditScore': [credit_score],
    'EstimatedSalary': [estimated_salary],
    'Tenure': [tenure],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]

})

##onehot encode the geography column
geo_encoded=onehotencoder_geo.transform([input_data[Geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehotencoder_geo.get_feature_names_out(['Geography']))

#concatenate the onehot encoded geography columns with the input dataframe
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#scaling the input data using the loaded scaler
input_scaled=scaler.transform(input_data)

#predict churn
prediction = model.predict(input_scaled)
prediction_probability = prediction[0][0]

if prediction_probability > 0.5:
    st.write("The customer is likely to churn.")
if prediction_probability <= 0.5:
    st.write("The customer is not likely to churn.")

 


                         
