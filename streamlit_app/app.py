import streamlit as st
from PIL import Image
from pathlib import Path

from src.model.inference import predict

st.set_page_config(page_title="AI Crop Disease Detector", layout="centered")

st.title("AI Crop Disease Detector")
st.write("Upload a leaf image to get a prediction (placeholder model)")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"]) 
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded", use_column_width=True)
    with st.spinner("Predicting..."):
        result = predict(image)
    st.json(result)

# Optional: show where to place exported models
models_dir = Path("mobile_integration")
models_dir.mkdir(parents=True, exist_ok=True)
st.caption(f"Mobile model artifacts stored in: {models_dir.resolve()}")
