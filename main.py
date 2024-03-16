import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np

# Load the pre-trained model for pneumonia detection
model = tf.keras.models.load_model('pneumonia.h5')

st.title("Covid19 pnuemonia detection")
st.markdown("<h1 style='text-align: center; color: red;'>COVID-19 Pneumonia Diagnosis</h1>", unsafe_allow_html=True)
st.write("Pneumonia is a respiratory infection that inflames the air sacs in one or both lungs.")

# Subheader for Symptoms
st.subheader("Symptoms of Pneumonia")
st.markdown("""
- Cough
- Fever
- Chest pain
- Shortness of breath
- Fatigue
- Other symptoms may include chills, sweating, headache, muscle pain, and loss of appetite.
""")

st.set_option('deprecation.showfileUploaderEncoding', False)
img4 = st.file_uploader("Upload a chest X-ray image...", type=["jpg", "jpeg", "png"])

if img4 is not None:
    # Open the image using PIL and convert it to a NumPy array
    image = Image.open(img4)
    
    # Ensure the image is in RGB format (3 channels)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize the image to match the model's input shape
    image = image.resize((224, 224))
    
    # Convert the image to a NumPy array
    img = np.array(image)
    
    # Normalize pixel values to the range [0, 1]
    img = img / 255.0
    
    # Add a batch dimension to the image
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)

    # Map the class index to a label
    class_labels = ["Normal", "Pneumonia"]
    predicted_class_label = class_labels[predicted_class_index]

    st.image(image, caption='Uploaded X-ray Image', use_column_width=True)
    
    # Add a centered big heading for the prediction
    st.markdown(f"<h1 style='text-align: center; color: {'green' if predicted_class_label == 'Normal' else 'red'};'>{predicted_class_label}</h1>", unsafe_allow_html=True)
    
    if predicted_class_label == "Pneumonia":
        st.header("Treatment for Pneumonia")
        st.markdown("""
        - Antibiotics (e.g., Amoxicillin, Azithromycin)
        - Rest and hydration
        - Symptomatic relief (e.g., pain and fever relievers)
        - Oxygen therapy (if necessary)
        """)
    else:
        st.write("No pneumonia detected. Consult a healthcare professional for further evaluation.")
else:
    st.write("Please upload a chest X-ray image for diagnosis.")
