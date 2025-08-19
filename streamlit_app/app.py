# streamlit_app/app.py
import os, json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.applications.mobilenet_v3 import MobileNetV3Small, preprocess_input

# --------------------------
# Best hyperparameters (fixed)
# --------------------------
BEST = {
    "use_data_augmentation": False,
    "dense_units": 256,
    "dropout": 0.1,
    "activation": "swish",           # -> use glorot_uniform kernel init
}

# --------------------------
# Paths & constants
# --------------------------
IMG_SIZE = (160, 160)
WEIGHTS_PATH = "model/best_model_tuned.weights.h5"
CLASS_NAMES_PATH = "model/class_names.json"


# --------------------------
# Utilities
# --------------------------
@st.cache_resource(show_spinner=False)
def load_class_names(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            names = json.load(f)
        if isinstance(names, list) and all(isinstance(x, str) for x in names):
            return names
        raise ValueError("class_names.json must be a JSON list of strings.")
    st.warning("class_names.json not found; labels will be shown as 'Class i'.")
    return None

def build_inference_model(num_classes: int, img_size=IMG_SIZE) -> tf.keras.Model:
    """
    Functional model for inference that mirrors the training head:
      Input -> MobileNetV3Small (no top) -> GAP -> Dropout(0.1)
            -> Dense(256, no bias, BN, swish) -> Dense(num_classes, softmax)
    """
    base = MobileNetV3Small(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights=None  # weights will come from .weights.h5
    )
    base.trainable = False

    inp = L.Input(shape=(img_size[0], img_size[1], 3), name="input")
    x = base(inp)  # don't pass training=...
    x = L.GlobalAveragePooling2D(name="gap")(x)
    x = L.Dropout(BEST["dropout"], name="dropout")(x)

    # kernel init to match swish
    kernel_init = "glorot_uniform" if BEST["activation"] == "swish" else "he_normal"

    # Dense head: no bias -> BN -> activation(swish)
    x = L.Dense(
        BEST["dense_units"],
        use_bias=False,
        kernel_initializer=kernel_init,
        name=f"dense{BEST['dense_units']}"
    )(x)
    x = L.BatchNormalization(name="bn")(x)
    act = tf.keras.activations.swish if BEST["activation"] == "swish" else tf.keras.activations.relu
    x = L.Activation(act, name=BEST["activation"])(x)

    out = L.Dense(num_classes, activation="softmax", dtype="float32", name="pred")(x)
    return tf.keras.Model(inp, out, name="leaf_inference")

def preprocess_image_pil(pil_img: Image.Image, img_size=IMG_SIZE) -> np.ndarray:
    """Resize + MobileNetV3 preprocess_input (expects 0..255 RGB). Returns (1,H,W,3) float32."""
    img = pil_img.convert("RGB").resize(img_size)
    arr = np.asarray(img, dtype=np.float32)
    arr = preprocess_input(arr)  # maps to [-1, 1]; do NOT divide by 255
    return np.expand_dims(arr, axis=0)


@st.cache_resource(show_spinner=True)
def load_model_and_labels():
    class_names = load_class_names(CLASS_NAMES_PATH)
    if class_names is None:
        # You can still run, but labels will be generic
        st.info("Tip: add class_names.json for friendly labels.")

    num_classes = len(class_names) if class_names else 2  # fallback guess
    model = build_inference_model(num_classes=num_classes, img_size=IMG_SIZE)

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights file not found: {WEIGHTS_PATH}")
    model.load_weights(WEIGHTS_PATH)

    return model, class_names


# --------------------------
# UI
# --------------------------
st.title("🌿 AI Crop Disease Detector")
st.write("Upload a crop leaf image to detect diseases using AI")

try:
    model, class_names = load_model_and_labels()
    st.success("✅ Model loaded (weights)")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload a crop leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        x = preprocess_image_pil(image, IMG_SIZE)
        pred = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(pred))
        conf = float(np.max(pred)) * 100.0

        # label
        name = class_names[idx] if class_names and idx < len(class_names) else f"Class {idx}"
        st.success(f"🔍 Prediction: **{name}**")
        st.info(f"📊 Confidence: **{conf:.2f}%**")

        st.write("**All Class Probabilities:**")
        for i, p in enumerate(pred):
            cname = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
            st.write(f"- {cname}: {p*100:.2f}%")
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
