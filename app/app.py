
import streamlit as st
import pickle
import numpy as np

# Load the trained model and pipeline
model = pickle.load(open('mental_health_prediction_model.pkl', 'rb'))
pipeline = pickle.load(open('preprocessing_pipeline.pkl', 'rb'))

st.title('Mental Health Support Prediction')

# User input for the prediction
age = st.slider('Age', 18, 65, 25)
gender = st.selectbox('Gender', ['Male', 'Female'])
work_interfere = st.selectbox('Work Interfere', ['Never', 'Rarely', 'Sometimes', 'Often'])
benefits = st.selectbox('Benefits', ['Yes', 'No'])

# Predict button
if st.button('Predict'):
    input_data = np.array([[age, gender, work_interfere, benefits]])
    input_data_transformed = pipeline.transform(input_data)
    prediction = model.predict(input_data_transformed)
    
    if prediction[0] == 1:
        st.success('The person is likely to seek mental health support.')
    else:
        st.error('The person is not likely to seek mental health support.')
