from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Initialize label encoder and fit it with the labels
label = ['no', 'yes']
le = LabelEncoder()
le.fit_transform(label)

# Load the trained model
model = load_model("model.h5")

# Streamlit app title
st.title("Brain Tumor Detection")

# Upload file widget
file = st.file_uploader("Choose an image file")

if file is not None:
    # Load and display the image
    img = load_img(file, target_size=(69, 69))
    st.image(img, caption="Uploaded Image")

    # Convert image to array and preprocess
    array = img_to_array(img)
    img_array = np.expand_dims(array, axis=0)
    img_array /= 255.0

    # Predict the class of the image
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    
    # Decode the predicted class to label
    pred_label = le.inverse_transform(predicted_class)
    
    # Display the prediction
    st.write(f"Prediction: {pred_label[0]}")