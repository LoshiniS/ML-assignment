import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model_path = "model.pkl"  # Ensure this file is in the working directory
model = joblib.load(model_path)

# Define the input fields based on dataset features
feature_names = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea',
                 'EquivDiameter', 'Extent', 'Solidity', 'Roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2',
                 'ShapeFactor3', 'ShapeFactor4']

st.title("Dry Bean Classification")
st.write("Enter the feature values to predict the bean type:")

# Create input fields
user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0, format="%.4f")
    user_input.append(value)

# Prediction button
if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    st.write(f"Input shape: {input_array.shape}")  # Debugging output
    prediction = model.predict(input_array)
    st.success(f"Predicted Bean Type: {prediction[0]}")
