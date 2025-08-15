import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="YouLarva",
    page_icon="🐛",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------- JSON FILE FOR COUNT -------------------
COUNT_FILE = "counts.json"

# Initialize JSON if not exists
if not os.path.exists(COUNT_FILE):
    with open(COUNT_FILE, "w") as f:
        json.dump({"Chapri": 0, "Decent": 0}, f)

# Functions to read and update counts
def read_counts():
    with open(COUNT_FILE, "r") as f:
        return json.load(f)

def update_count(label):
    counts = read_counts()
    counts[label] += 1
    with open(COUNT_FILE, "w") as f:
        json.dump(counts, f)

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("chapri_decent.h5")

model = load_model()
_, height, width, channels = model.input_shape
img_size = (width, height)

# ------------------- MAIN TITLE -------------------
st.markdown("<h1 style='text-align:center; color: #4CAF50;'>YouLarva</h1>", unsafe_allow_html=True)
st.write("### Upload an image to get started:")

# ------------------- FILE UPLOAD -------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

# ------------------- PREDICTION -------------------
if uploaded_file is not None:
    # Read current counts
    counts = read_counts()

    col1, col2 = st.columns([1, 1.2])  # Left: Image | Right: Prediction

    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=250)

    with col2:
        img = image.resize(img_size)
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        prediction = model.predict(img_array)[0][0]
        class_label = "Decent" if prediction > 0.5 else "Chapri"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        # Update JSON counts
        update_count(class_label)

        st.markdown(f"### Prediction: **{class_label}**")
        st.progress(float(confidence))
        st.write(f"**Confidence:** {confidence:.2%}")

        # Refresh counts in center
        counts = read_counts()
        st.markdown(
            f"<h4 style='text-align:center; color:#fa003f'>Chapri: {counts['Chapri']} | Decent: {counts['Decent']}</h4>",
            unsafe_allow_html=True
        )

        # ------------------- EMOJI RAIN PARALLAX -------------------
        st.markdown(
            """
            <style>
            .emoji {
                position: fixed;
                top: -50px;
                pointer-events: none;
                opacity: 0.8;
                animation-name: fall;
                animation-iteration-count: infinite;
                animation-timing-function: linear;
                z-index: 9999;
            }

            @keyframes fall {
                0% { transform: translateY(0px); }
                100% { transform: translateY(110vh); }
            }
            </style>
            <script>
            const emojis = ["🐛","🦋","🌱","🍃","💚","🐞","🪱"];
            const numEmojis = 50;

            function random(min, max) {
                return Math.random() * (max - min) + min;
            }

            for(let i=0; i<numEmojis; i++) {
                let span = document.createElement("span");
                span.className = "emoji";
                span.innerText = emojis[Math.floor(Math.random() * emojis.length)];
                document.body.appendChild(span);

                // Random horizontal position
                span.style.left = random(0, 100) + "vw";

                // Random size for parallax effect
                const size = random(15, 50);
                span.style.fontSize = size + "px";

                // Random speed
                const duration = random(3, 10);
                span.style.animationDuration = duration + "s";

                // Random delay
                span.style.animationDelay = random(0, 10) + "s";
            }
            </script>
            """,
            unsafe_allow_html=True
        )

# ------------------- FOOTER -------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray; font-weight:bold;'>Developed with <span style='color:#fa003f'>Love & Larva</span></p>",
    unsafe_allow_html=True
)
