import streamlit as st
import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

st.set_page_config(page_title="Number Plate Detection", layout="wide")
st.title("🚗 Number Plate Detection App")

# Upload image
uploaded_file = st.file_uploader("Upload an image of a car", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = np.array(Image.open(uploaded_file))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (1000, 700))
    output_image = image.copy()

    st.subheader("Original Image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)
    st.subheader("Edge Detection")
    st.image(edges, use_column_width=True, clamp=True)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Initialize EasyOCR
    reader = easyocr.Reader(['en'])
    plate_count = 0
    detected_texts = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        if 2 < aspect_ratio < 6 and 100 < w < 600 and 30 < h < 200:
            roi = image[y:y+h, x:x+w]
            result = reader.readtext(roi)

            for detection in result:
                text = detection[1].strip()
                if len(text) >= 5:
                    plate_count += 1
                    detected_texts.append(text)

                    cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(output_image, f'Plate #{plate_count}: {text}',
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    break

    if plate_count > 0:
        st.subheader("Detected Number Plates")
        st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), use_column_width=True)
        st.success(f"Detected Plates: {', '.join(detected_texts)}")
    else:
        st.warning("No number plates detected!")
