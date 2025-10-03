import streamlit as st
import joblib
from PIL import Image
import numpy as np
import cv2

# --- Load model ---
with open("svm_image_classifier_model.pkl", "rb") as f:
    model = joblib.load(f)

# --- UI ---
st.title("Fruit Classifier")
st.write(" ")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# ตรวจสอบ class_dict ให้ตรงกับที่เทรนจริง
class_dict = {0: "Apple", 1: "Orange"}

if uploaded_file is not None:
    # แสดงรูปภาพที่อัปโหลด
    image = Image.open(uploaded_file).convert('RGB')  # แปลงเป็น RGB ให้ชัดเจน
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Predict"):
        # --- Preprocess ภาพ ---
        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        image_resized = cv2.resize(image_array, (100, 100))  # ขนาดต้องตรงกับตอนเทรน

        # Flatten เป็น 1D array
        image_flatten = image_resized.flatten().reshape(1, -1)

        # --- ทำนายผล ---
        prediction = model.predict(image_flatten)[0]
        prediction_name = class_dict.get(prediction, "Unknown")
        st.write(f"Prediction: **{prediction_name}**")
