import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Binary CNN Classifier",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("chapri_decent.h5")  # Your trained model

model = load_model()

# Detect model input size dynamically
_, height, width, channels = model.input_shape
img_size = (width, height)

# ------------------- SIDEBAR -------------------
# st.sidebar.title("📌 About App")
# st.sidebar.info(
#     """
#     **Binary Image Classifier**  
#     Upload an image to classify it into **Class 0** or **Class 1**.  
#     Model trained using CNN.
#     """
# )
# st.sidebar.write("Confidence score will be displayed after prediction.")

# ------------------- MAIN TITLE -------------------
st.markdown("<h1 style='color: #4CAF50;'>Binary Image Classifier</h1>", unsafe_allow_html=True)
st.write("### Upload an image to get started:")

# ------------------- FILE UPLOAD -------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

# ------------------- PREDICTION -------------------
if uploaded_file is not None:
    # Columns for side-by-side display
    col1, col2 = st.columns([1, 1.2])  # Left: Image | Right: Prediction

    with col1:
        # Show uploaded image (small size)
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=250)

    with col2:
        # Preprocess for model
        img = image.resize(img_size)
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)  # Normalize

        # Predict
        prediction = model.predict(img_array)[0][0]
        class_label = "Decent" if prediction > 0.5 else "Chapri"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        # Display results neatly
        st.markdown(f"### Prediction: **{class_label}**")
        st.progress(float(confidence))
        st.write(f"**Confidence:** {confidence:.2%}")

# ------------------- FOOTER -------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>Developed with ❤️ using Streamlit & TensorFlow</p>",
    unsafe_allow_html=True
)
