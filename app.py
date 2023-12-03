import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model

# Load the trained model
model = load_model("./tf_digit_classifier.h5")  # Replace with your model path

def preprocess_image(img):
    if len(np.shape(img)) == 2:
        img = np.expand_dims(img, axis=-1)
    else:
        img = tf.image.rgb_to_grayscale(img)  # Convert to grayscale

    print("after", np.shape(img))
    img = tf.image.resize(img, (28, 28))   # Resize to model input size
    img = img / 255.0                      # Normalize pixel values
    img = np.expand_dims(img, axis=0)      # Add batch dimension
    return img

# Function to make predictions
def predict_image(img):
    processed_img = preprocess_image(np.array(img))
    prediction = model.predict(processed_img)
    st.image(image, caption='Uploaded Image.', width=256)
    st.write("Classifying image..")
    st.write("")
    predicted_class = np.argmax(prediction)
    return predicted_class

st.title("Handwritten Digit Recognition")
uploaded_file = st.file_uploader("Choose an image (of type JPG)", type=["png", "jpg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    predicted_class = predict_image(image)
    st.write(f"Prediction: {predicted_class}")
