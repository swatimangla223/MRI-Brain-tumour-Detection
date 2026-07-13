import streamlit as st
from train_model import HOG_ORIENTATIONS



#page config- Must be first
st.set_page_config(page_title="Brain Tumor Classification", layout="centered")


#imports (after config)

import cv2
import numpy as np
from skimage.feature import hog
import joblib
import os

# Load Saved Artifacts

@st.cache_resource
def load_artifacts():
    model_path = joblib.load("decision_tree_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    return model_path, scaler, le

#Try to load model;; show error if files are missing
try:
    model, scaler, le = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error("Model files not found. Please run 'train_model.py' first to generate the required'pkl' files")

# CONFIGURATIONS

IMG_SIZE = (128, 128)  # Resize images to this size
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

def preprocess_image(image):
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # Resize image
    resized = cv2.resize(gray, IMG_SIZE)
    # HOG features
    hog_feat = hog(resized, orientations=HOG_ORIENTATIONS, 
                   pixels_per_cell=HOG_PIXELS_PER_CELL, 
                   cells_per_block=HOG_CELLS_PER_BLOCK,
                   transform_sqrt=True , block_norm='L2-Hys')
    return hog_feat

#Streamlit UI

st.title("Brain Tumor Classification")
st.markdown("Upload an MRI scan image and the model will predict the tumour type")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption='Uploaded Image', use_column_width=True)

    with st.spinner('Analyzing...'):
        # Preprocess
        features = preprocess_image(image)
        features_scaled = scaler.transform([features])

        # Predict
        pred_label_enc = model.predict(features_scaled)[0]
        pred_class = le.inverse_transform([pred_label_enc])[0]

        # Get Probability (if supported)
        proba = model.predict_proba(features_scaled)[0]
        confidence = np.max(proba) * 100

        # Show result
        st.success(f"Prediction: {pred_class}")
        st.metric("Confidence", f"{confidence:.2f}%")

        # Show probability breakdown
        st.write("Class Probabilities:")
        for cls, prob in zip(le.classes_, proba):
            st.write(f"{cls}: {prob*100:.2f}%")
elif uploaded_file is not None and not model_loaded:
    st.warning("Cannot make predictions as model files are missing.")
else:
    st.info("Please upload an MRI image to get a prediction.")