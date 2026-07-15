import streamlit as st
import google.generativeai as genai
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="AI Object Detection")

st.title("Object Detection using YOLO + Gemini AI")

api_key = st.text_input("Enter Gemini API Key", type="password")

uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded is not None and api_key:

    image = Image.open(uploaded)

    st.image(image, caption="Uploaded Image")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-2.5-flash")

    yolo = YOLO("yolov8n.pt")

    results = yolo.predict(image)

    names = []

    for r in results:
        boxes = r.boxes

        for b in boxes:
            cls = int(b.cls[0])
            names.append(yolo.names[cls])

    detected = list(set(names))

    st.subheader("Detected Objects")

    st.write(detected)

    response = model.generate_content(
        f"""
        The detected objects are:

        {detected}

        Explain these objects briefly and describe what is happening in the image.
        """
    )

    st.subheader("Gemini AI Explanation")

    st.write(response.text)

    plotted = results[0].plot()

    st.image(plotted, caption="Detection Result")