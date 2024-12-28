import streamlit as st
from Project_2_Titanic_Survivability_Prediction import model

st.title("Titanic Survivability Prediction :ship:")
st.title("By : Manas Ranjan Jena")

labels = {1 : 'Survived', 0 : 'Did not Survived'}

Pclass = st.select_slider('Passenger Class', [1, 2, 3])
Sex = st.selectbox('Gender', ['Male', 'Female'])
if Sex == 'Male':
    Sex = 0
else :
    Sex = 1
age = st.number_input('Age', min_value = 1, max_value = 100, step = 1)
SibSp = st.select_slider('Number of Siblings/Parents', [i for i in range(1, 11)])
Parch = st.select_slider('Parch', [i for i in range(0, 6)])

details = [[Pclass, Sex, age, SibSp, Parch]]

click = st.button('Predict')

if click :
    pred = model.predict(details)
    print(pred)
    label = labels[pred[0]]
    if pred == 0:
        emoji = 'skull_and_crossbones'
    else :
        emoji = 'smile'
    st.write(f'### The passenger {label} :{emoji}:')
