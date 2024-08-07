{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "4b3ddafb-4795-465e-84b1-39e6d1672454",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "# Load the trained model\n",
    "try:\n",
    "    model = joblib.load('logistic_regression_model.pkl')\n",
    "except Exception as e:\n",
    "    st.error(f'Error loading model: {e}')\n",
    "\n",
    "# Define the Streamlit app\n",
    "def main():\n",
    "    st.title(\"Titanic Survival Prediction\")\n",
    "    html_temp=\"\"\"\n",
    "    <div style=\"background-color:yellow;padding:13px\">\n",
    "    <h1 style=\"color:black;text-align:center;\">Streamlit Survive Prediction ML App</h1>\n",
    "    </div>\n",
    "    \"\"\"\n",
    "    st.markdown(html_temp,unsafe_allow_html=True)\n",
    "\n",
    "    # User inputs\n",
    "    pclass = st.selectbox('Passenger Class', [1, 2, 3])\n",
    "    age = st.number_input('Age', min_value=0, max_value=100, value=25)\n",
    "    sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)\n",
    "    parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)\n",
    "    fare = st.number_input('Fare', min_value=0.0, max_value=500.0, value=50.0)\n",
    "    sex = st.selectbox('Sex', ['male', 'female'])\n",
    "    embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])\n",
    "\n",
    "    # Encode inputs\n",
    "    sex_male = 1 if sex == 'male' else 0\n",
    "    embarked_C = 1 if embarked == 'C' else 0\n",
    "    embarked_Q = 1 if embarked == 'Q' else 0\n",
    "    embarked_S = 1 if embarked == 'S' else 0\n",
    "\n",
    "    # Predict button\n",
    "    if st.button('Predict'):\n",
    "        # Prepare input data\n",
    "        input_data = np.array([[pclass, age, sibsp, parch, fare, sex_male, embarked_C, embarked_Q, embarked_S]])\n",
    "        \n",
    "        try:\n",
    "            prediction = model.predict(input_data)[0]\n",
    "            probability = model.predict_proba(input_data)[0, 1]\n",
    "            \n",
    "            # Display prediction and probability\n",
    "            if prediction == 1:\n",
    "                st.success(f'The passenger is likely to survive with a probability of {probability:.2f}.')\n",
    "            else:\n",
    "                st.error(f'The passenger is not likely to survive with a probability of {1-probability:.2f}.')\n",
    "        except Exception as e:\n",
    "            st.error(f'Error during prediction: {e}')\n",
    "\n",
    "# Run the app\n",
    "if __name__ == '__main__':\n",
    "    main()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
