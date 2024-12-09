#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install streamlit


# In[ ]:


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf

# Title for Streamlit app
st.title("Tumor Classification AI")
st.write("Upload a medical image to classify the type of tumor or confirm healthy cells.")

# Load the model
@st.cache_resource  # Cache model to avoid reloading on every interaction
def load_model():
    return tf.keras.models.load_model('bestmodel2.keras')

model = load_model()

# Class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Function to predict and display tumor type
def predict_and_display(uploaded_file):
    img = load_img(uploaded_file, target_size=(256, 256))  # Resize to the model's expected input size
    input_arr = img_to_array(img) / 255.0  # Normalize to [0, 1]
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    predictions = model.predict(input_arr)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]

    # Display image and prediction using matplotlib
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Predicted Class: {predicted_class_label}", fontsize=14, color='blue')

    # Add detailed explanation based on tumor type
    if predicted_class_label == 'notumor':
        ax.set_xlabel("Healthy cells scan", fontsize=12, color='green')
    elif predicted_class_label == 'glioma':
        ax.set_xlabel("Glioma tumor", fontsize=12, color='red')
    elif predicted_class_label == 'meningioma':
        ax.set_xlabel("Meningioma tumor", fontsize=12, color='orange')
    elif predicted_class_label == 'pituitary':
        ax.set_xlabel("Pituitary tumor", fontsize=12, color='purple')

    st.pyplot(fig)

# File uploader in Streamlit
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image and predictions
    st.write("Processing the uploaded image...")
    predict_and_display(uploaded_file)

