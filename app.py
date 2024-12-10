#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import altair as alt

# Title for Streamlit app
st.title("MRI Scan Brain Tumor Classification AI")
st.write("Upload a MRI medical image to classify the type of Brain tumor or confirm healthy cells.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('bestmodel2.keras')

model = load_model()

# Updated class labels with descriptive names
class_labels = {
    'glioma': 'Glioma Tumor',
    'meningioma': 'Meningioma Tumor',
    'notumor': 'Healthy Cells Scan',
    'pituitary': 'Pituitary Tumor'
}

def predict_and_display(uploaded_file):
    img = load_img(uploaded_file, target_size=(256, 256))
    input_arr = img_to_array(img) / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    predicted_class_index = np.argmax(predictions)
    predicted_class = list(class_labels.keys())[predicted_class_index]
    
    # Get the descriptive label
    display_label = class_labels[predicted_class]

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Prediction: {display_label}", fontsize=14, color='blue')

    # Set color based on prediction
    label_colors = {
        'notumor': 'green',
        'glioma': 'red',
        'meningioma': 'orange',
        'pituitary': 'purple'
    }
    
    ax.set_xlabel(display_label, fontsize=12, color=label_colors[predicted_class])
    st.pyplot(fig)

# File uploader in Streamlit
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("Processing the uploaded image...")
    predict_and_display(uploaded_file)

