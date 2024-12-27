# Importing dependencies
import streamlit as st
from Project_Vehicle_Price_Prediction import model, label_encode, data, categorical_colmns

st.title('Vehicle Price Prediction Model :car:')
st.title("By : Manas Ranjan Jena")

name = st.text_input('Enter the name of the vehicle')
description = st.text_input('Enter a description of the vehicle')
make = st.text_input('Enter the materials it is made of')
models = st.text_input('Enter the model of the vehicle')
year = st.number_input('Enter the release year of the vehicle')
engine = st.text_input('Enter the engine of the vehicle')
cylinders = st.number_input('Enter the number of cylinders in the vehicle')
fuel = st.text_input('Enter the fuel capacity in Litres')
mileage = st.number_input('Enter the mileage provided by the vehicle')
transmission = st.text_input('Enter the transmission of the vehicle')
trim = st.text_input('Enter the trim of the vehicle')
body = st.text_input('Enter a brief description about the body of the vehicle')
doors = st.number_input('Enter the number of doors in the vehicle')
exterior_color = st.text_input('Enter the exterior color of the vehicle')
interior_color = st.text_input('Enter the interior color of the vehicle')
drivetrain = st.text_input('Enter the drivetrain of the vehicle')

details = [name, description, make, models, year, engine, cylinders, fuel, mileage, transmission, trim, body, doors, exterior_color, interior_color, drivetrain]

click = st.button('Predict Price')

if click :
    processed_details = label_encode.transform(details)
    price = model.predict(processed_details)
    st.write(f"### The price of the vehicle is estimated to be {price}")

