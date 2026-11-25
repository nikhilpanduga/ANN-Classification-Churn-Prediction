
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# --------------------------------
# Load trained model
# --------------------------------

model = tf.keras.models.load_model('model.h5')

# --------------------------------
# Load encoders and scaler
# --------------------------------

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehotencoder_geo.pkl', 'rb') as f:
    onehotencoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# --------------------------------
# Streamlit application
# --------------------------------

st.title("Customer Churn Prediction")

st.write("Enter customer details to predict whether the customer is likely to churn.")

# --------------------------------
# User input
# --------------------------------

Geography = st.selectbox(
    "Geography",
    onehotencoder_geo.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.number_input("Age", min_value=18, max_value=92, value=30)

balance = st.number_input("Balance", min_value=0.0, value=0.0)

credit_score = st.number_input("CreditScore", min_value=0, value=600)

estimated_salary = st.number_input(
    "EstimatedSalary",
    min_value=0.0,
    value=50000.0
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

num_of_products = st.number_input(
    "NumberOfProducts",
    min_value=1,
    max_value=4,
    value=1
)

has_cr_card = st.selectbox("Has CreditCard", [0, 1])

is_active_member = st.selectbox("IsActiveMember", [0, 1])


# --------------------------------
# Prediction button
# --------------------------------

if st.button("Predict Churn"):

    try:

        # Encode Gender
        gender_encoded = label_encoder_gender.transform([gender])[0]

        # Prepare numerical input data
        input_data = pd.DataFrame({
            'Gender': [gender_encoded],
            'Age': [age],
            'Balance': [balance],
            'CreditScore': [credit_score],
            'EstimatedSalary': [estimated_salary],
            'Tenure': [tenure],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member]
        })

        # --------------------------------
        # One-hot encode Geography
        # --------------------------------

        geo_encoded = onehotencoder_geo.transform(
            [[Geography]]
        ).toarray()

        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=onehotencoder_geo.get_feature_names_out(
                ['Geography']
            )
        )

        # --------------------------------
        # Combine all input features
        # --------------------------------

        input_data = pd.concat(
            [input_data.reset_index(drop=True),
             geo_encoded_df.reset_index(drop=True)],
            axis=1
        )

        # --------------------------------
        # Match scaler training columns
        # --------------------------------

        expected_columns = scaler.feature_names_in_

        missing_columns = set(expected_columns) - set(input_data.columns)

        if missing_columns:
            st.error(f"Missing columns: {missing_columns}")
            st.stop()

        input_data = input_data[expected_columns]

        # --------------------------------
        # Scale input data
        # --------------------------------

        input_scaled = scaler.transform(input_data)

        # --------------------------------
        # Predict customer churn
        # --------------------------------

        prediction = model.predict(input_scaled, verbose=0)

        prediction_probability = float(prediction[0][0])

        # --------------------------------
        # Display prediction
        # --------------------------------

        st.subheader("Prediction Result")

        st.metric(
            "Churn Probability",
            f"{prediction_probability * 100:.2f}%"
        )

        if prediction_probability > 0.5:

            st.error("The customer is likely to churn.")

        else:

            st.success("The customer is not likely to churn.")

    except Exception as e:

        st.error(f"An error occurred: {e}")
