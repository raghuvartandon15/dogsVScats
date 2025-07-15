import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

model=tf.keras.models.load_model('best_model.h5')

def preprocess_img(uploaded_file):
    img=Image.open(uploaded_file).convert("L")
    img=np.array(img)
    img=cv2.resize(img, (64, 64))
    img=img/255.0
    img=img.reshape(1, 64, 64, 1)
    return img

def predict_img(uploaded_file):
    img=preprocess_img(uploaded_file)
    prediction=model.predict(img)[0][0]
    if prediction>0.5:
        st.success(f"🐶 Prediction: **Dog** ({prediction:.2f})")
    else:
        st.success(f"🐱 Prediction: **Cat** ({1 - prediction:.2f})")
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

st.title('🐶🐱 Dog vs Cat Predictor')
st.header("Upload an image of a **cat** or **dog**")
uploaded_file=st.file_uploader(label="📁 Upload Image", accept_multiple_files=False, type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    if st.button("🔍 Predict"):
        predict_img(uploaded_file)
    