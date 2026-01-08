import streamlit as st
import cv2
import numpy as np
from PIL import Image

# App title
st.title("Human Face Identification App")
st.write("Upload an image and the app will identify human faces.")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # Draw rectangles and label
    for (x, y, w, h) in faces:
        cv2.rectangle(
            img_array, (x, y), (x + w, y + h), (0, 255, 0), 2
        )
        cv2.putText(
            img_array,
            "Human face identified",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Display results
    st.subheader("Processed Image")
    st.image(img_array, channels="BGR")

    st.success(f"Total Faces Detected: {len(faces)}")
