import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('iris_classifier_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the class names
class_names = ['Setosa', 'Versicolor', 'Virginica']

def predict_species(features):
    # Standardize the features
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features)
    predicted_species = class_names[prediction[0]]
    return predicted_species

# Streamlit app
st.title("Iris Species Classifier")
st.write("Enter the features of the Iris flower to classify its species:")

# Input fields for the features
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=1.0)

# Button to make the prediction
if st.button('Predict'):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    species = predict_species(features)
    st.write(f'The predicted species is: **{species}**')
