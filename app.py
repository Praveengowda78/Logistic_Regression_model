import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model, scaler, and metrics
try:
    model = joblib.load('logistic_regression_model.pkl')
    scaler = joblib.load('scaler.pkl')
    metrics = joblib.load('metrics.pkl')
except Exception as e:
    st.error(f'Error loading model, scaler, or metrics: {e}')

st.title("Titanic Survival Prediction")
html_temp = """
<div style="background-color:yellow;padding:13px">
<h1 style="color:black;text-align:center;">Streamlit Survival Prediction ML App</h1>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# User inputs
pclass = st.selectbox('Passenger Class', [1, 2, 3])
age = st.number_input('Age', min_value=0, max_value=100, value=25)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)
fare = st.number_input('Fare', min_value=0.0, max_value=500.0, value=50.0)
sex = st.selectbox('Sex', ['male', 'female'])
Embarked = st.selectbox('Port of Embarkation', [ 'Q', 'S'])

# Encode inputs
sex_male = 1 if sex == 'male' else 0

Embarked_Q = 1 if Embarked == 'Q' else 0
Embarked_S = 1 if Embarked == 'S' else 0

# Prepare input data in the same format as training data
input_data = pd.DataFrame([[pclass, age, sibsp, parch, fare, sex_male, Embarked_Q, Embarked_S]],
                          columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S'])

# Check for missing columns and add them with default value 0
expected_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns to match the order used during model training
input_data = input_data[expected_columns]

# Scale the input data
input_data_scaled = scaler.transform(input_data)

if st.button('Predict'):
    try:
        prediction = model.predict(input_data_scaled)[0]
        probability = model.predict_proba(input_data_scaled)[0, 1]

        # Display prediction and probability
        if prediction == 1:
            st.success(f'The passenger is likely to survive with a probability of {probability:.2f}.')
        else:
            st.error(f'The passenger is not likely to survive with a probability of {1-probability:.2f}.')
    except Exception as e:
        st.error(f'Error during prediction: {e}')

# Display model metrics
st.write(f"Model Accuracy: {metrics['accuracy'] * 100:.2f}%")
st.write(f"Precision: {metrics['precision']:.2f}")
st.write(f"Recall: {metrics['recall']:.2f}")
st.write(f"F1-Score: {metrics['f1']:.2f}")
st.write(f"ROC-AUC Score: {metrics['roc_auc']:.2f}")
