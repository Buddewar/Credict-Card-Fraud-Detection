import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Loading the model
model_path = r"randomForest_model.pkl"
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict the entered values in GUI
def predict_default(features):
    # Preprocess the features
    features_array = np.array(features).reshape(1, -1)
    # Predicting
    prediction = model.predict(features_array)
    return prediction

st.title('Credit Card Fraud Detection')
st.write('Enter the values of the features to predict if there will be a default payment next month.')

# Input fields for features
LIMIT_BAL = st.number_input('LIMIT_BAL')
SEX = st.selectbox('SEX', ['Male', 'Female'])
EDUCATION = st.selectbox('EDUCATION', ['Graduate School', 'University', 'High School', 'Others'])
MARRIAGE = st.selectbox('MARRIAGE', ['Married', 'Single', 'Others'])
AGE = st.number_input('AGE')
PAY_0 = st.selectbox('PAY_0', [-1, 0, -2, 1, 2, 3, 4, 8])
PAY_2 = st.selectbox('PAY_2', [0, -1, -2, 2, 3, 5, 7, 4, 1])
PAY_3 = st.selectbox('PAY_3', [-1, 0, 2, -2, 3, 4, 6, 7, 1, 5])
PAY_4 = st.selectbox('PAY_4', [0, -2, -1, 2, 3, 4, 5, 7])
PAY_5 = st.selectbox('PAY_5', [0, -1, 2, -2, 3, 5, 4, 7])
PAY_6 = st.selectbox('PAY_6', [0, -1, 2, -2, 3, 6, 4, 7])
BILL_AMT1 = st.number_input('BILL_AMT1')
BILL_AMT2 = st.number_input('BILL_AMT2')
BILL_AMT3 = st.number_input('BILL_AMT3')
BILL_AMT4 = st.number_input('BILL_AMT4')
BILL_AMT5 = st.number_input('BILL_AMT5')
BILL_AMT6 = st.number_input('BILL_AMT6')
PAY_AMT1 = st.number_input('PAY_AMT1')
PAY_AMT2 = st.number_input('PAY_AMT2')
PAY_AMT3 = st.number_input('PAY_AMT3')
PAY_AMT4 = st.number_input('PAY_AMT4')
PAY_AMT5 = st.number_input('PAY_AMT5')
PAY_AMT6 = st.number_input('PAY_AMT6')
# Map categorical values to numerical
SEX = 1 if SEX == 'Male' else 2
EDUCATION_MAP = {'Graduate School': 1, 'University': 2, 'High School': 3, 'Others': 4}
EDUCATION = EDUCATION_MAP[EDUCATION]
MARRIAGE_MAP = {'Married': 1, 'Single': 2, 'Others': 3}
MARRIAGE = MARRIAGE_MAP[MARRIAGE]
# Making prediction
features = [LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1,
            BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5,
            PAY_AMT6]

if st.button('Predict'):
    prediction = predict_default(features)
    if prediction == 1:
        st.write('Prediction: Fraudulent Credit Card')
    else:
        st.write('Prediction: Not Fraudulent Credit Card')
    st.write('Prediction:', prediction)


# Create a DataFrame with the input values
input_df = pd.DataFrame([features], columns=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 
                                             'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 
                                             'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 
                                             'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'])


