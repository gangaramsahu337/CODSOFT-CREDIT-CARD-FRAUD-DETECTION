import streamlit as st
import pandas as pd
import pickle
from math import radians, sin, cos, sqrt, atan2

# Load the pipeline (including preprocessor and model)
with open('fraud_detection_model_xgb.pkl', 'rb') as f:
    model = pickle.load(f)

# Feature input function
def user_input_features():
    st.markdown("### Transaction Details")
    
    merchant = st.text_input('Merchant', 'fraud_Rippin, Kub and Mann')
    category = st.selectbox('Category', ['misc_net', 'grocery_pos', 'grocery_net','entertainment','gas_transport', 'misc_pos','personal_care','health_fitness','travel','shopping_net','shopping_pos', 'kids_pets','food dining'])
    amt = st.number_input('Amount', value=4.97)
    city = st.text_input('City', 'Moravian Falls')
    state = st.text_input('State', 'NC')
    city_pop = st.number_input('City Population', value=3495)
    age = st.number_input('Age', value=30)
    
    data = {
        'merchant': merchant,
        'category': category,
        'amt': amt,
        'city': city,
        'state': state,
        'city_pop': city_pop,
        'age': age,
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

st.title('Credit Card Fraud Detection')
st.write('Enter transaction details to predict if it is fraudulent.')

input_df = user_input_features()

if st.button('Predict'):
    try:
        # Predict using the loaded model
        prediction = model.predict(input_df)

        st.subheader('Prediction')
        fraud_result = 'Fraudulent' if prediction[0] == 1 else 'Not Fraudulent'
        if fraud_result == 'Fraudulent':
            st.error(fraud_result)
        else:
            st.success(fraud_result)
    except ValueError as e:
        st.error("Error during prediction:")
        st.error(e)
