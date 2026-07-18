import streamlit as st
from PIL import Image
import torch

from src.predict import run_edge_inference 

st.title("Wildlife Species Classifier")
st.write("Upload an image of a Bullfrog or Cane Toad, and I'll classify it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    prediction, _ = run_edge_inference("temp.jpg")

    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.success(f"The model predicts this is a: {prediction.upper()}")