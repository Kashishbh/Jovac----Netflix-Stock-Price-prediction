import streamlit as st
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PowerTransformer

# Load the saved transformer and model
with open('transformer_no_volume.pkl', 'rb') as transformer_file:
    transformer = pickle.load(transformer_file)

with open('model_lin.pkl', 'rb') as model_file:
    model_rand = pickle.load(model_file)

# Define input fields for user data (assuming 'open', 'high', 'low', 'close', 'adj close', 'year', 'month', 'day')
st.title("Stock Price Prediction")

open_price = st.number_input('Open Price', min_value=0.0)
high_price = st.number_input('High Price', min_value=0.0)
low_price = st.number_input('Low Price', min_value=0.0)
close_price = st.number_input('Close Price', min_value=0.0)
adj_close_price = st.number_input('Adj Close Price', min_value=0.0)
year = st.number_input('Year', min_value=2000, max_value=2050)
month = st.number_input('Month', min_value=1, max_value=12)
day = st.number_input('Day', min_value=1, max_value=31)

# Create a dataframe from user input
input_data = pd.DataFrame({
    'high': [high_price],
    'low': [low_price],
    'year': [year],
    'month': [month],
    'day': [day]
})

# Apply the same transformation as during training
transformed_input = transformer.transform(input_data)

# Make predictions using the Random Forest model
if st.button('Predict'):
    prediction = model_rand.predict(transformed_input)
    st.write(f'Predicted Price: {prediction[0]*100000}')
