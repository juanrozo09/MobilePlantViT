# streamlit_app/app.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
# model = tf.keras.models.load_model("mobile_integration/tflite_model.tflite", compile=False)  # or your .keras if needed

# For .keras
model = tf.keras.models.load_model("model/best_model_tuned.keras", compile=True)

# Load class names from the model
class_names = model.class_names

# Streamlit UI
st.title("🌿 AI Crop Disease Detector")
uploaded_file = st.file_uploader("Upload a crop leaf image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((160, 160))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    prediction = model.predict(img_array)
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {pred_class} ({confidence:.2f}%)")
