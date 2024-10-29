import pickle
import json
import os
import streamlit as st
from PIL import Image
from utils import preprocess_image, expected_celebrities

# Load the model and dictionary
with open('./model/model.pickle', 'rb') as file:
    model = pickle.load(file)

with open('./model/class_dictionary.json', 'r') as file:
    dictionary = json.load(file)

# Reverse dictionary for prediction label
r_dictionary = {v: k for k, v in dictionary.items()}

def predict_celeb(img):
    try:
        X = preprocess_image(img)
        index = model.predict(X)[0]
        celebrity = r_dictionary.get(index, "Unknown Celebrity")
        return celebrity
    except:
        return None



# Streamlit app
st.title("Sport Celebs Classifier")

st.subheader("Expected Celebrities")

# Create a row to display all celebrity images
expected_celebrities(dictionary)


# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)

    cols = st.columns(3)
    cols[0].image(image, caption='Uploaded Image', use_column_width=True)

    # Predict button
    if st.button("Predict Celebrity"):
        prediction = predict_celeb(image)
        if prediction:
            st.write(f"**Predicted Celebrity:** { prediction.replace('_', ' ').title() }")
        else:
            st.error("Prediction returned no result.")
