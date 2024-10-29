import cv2
import pywt
import os
import numpy as np
from PIL import Image
import streamlit as st

face_cascade = cv2.CascadeClassifier('./model/opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./model/opencv/haarcascades/haarcascade_eye.xml')


def get_cropped_image_if_2_eyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
    return None

def w2d(img, mode='haar', level=1):
    imArray = img
    # Convert to grayscale if the image has 3 channels
    if len(imArray.shape) == 3:
        imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    # Compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # Reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H

def preprocess_image(img):
    img_array = np.array(img.convert('RGB'))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Detect face and crop image if face with two eyes is found
    # Pass OpenCV-compatible format to next function
    cropped_image = get_cropped_image_if_2_eyes(img_cv)

    if cropped_image is None:
        print("Face with two eyes not detected in the image.")
        return None

    # Apply wavelet transformation
    wavelet_image = w2d(cropped_image)

    # Resize the images to 32x32
    scalled_raw_img = cv2.resize(cropped_image, (32, 32))
    scalled_img_har = cv2.resize(wavelet_image, (32, 32))

    # Combine the raw and wavelet-transformed images into a single feature vector
    combined_img = np.vstack((scalled_raw_img.reshape(32*32*3, 1), scalled_img_har.reshape(32*32, 1)))

    # Prepare the data in the correct format for prediction
    X = np.array([combined_img]).reshape(1, 4096).astype(float)

    return X


def expected_celebrities(dictionary):
    cols = st.columns(len(dictionary))
    for idx, (celeb_name, _) in enumerate(dictionary.items()):
        base_path = './celeb_images/'
        img_path = f"{base_path}{celeb_name}.png"
        if os.path.exists(img_path):
            img = Image.open(img_path)
            cols[idx].image(img.resize((60, 60)), 
                            caption=celeb_name.replace('_', ' ').title(), 
                            use_column_width=True)
