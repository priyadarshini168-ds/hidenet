import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(page_title="HideNet", page_icon="🔐", layout="wide")

IMG_SIZE = 256

@st.cache_resource
def load_models():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.hidenet import build_hidenet
    from src.revealnet import build_revealnet
    from src.detector import build_detector

    hidenet = build_hidenet()
    revealnet = build_revealnet()
    detector = build_detector()

    hidenet.load_weights('models/hidenet.h5')
    revealnet.load_weights('models/revealnet.h5')
    detector.load_weights('models/detector.h5')

    return hidenet, revealnet, detector

def preprocess(image):
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def postprocess(tensor):
    img = np.array(tensor[0])
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)

st.title("HideNet — Adversarial Image Steganography")
st.markdown("**Attention-Based U-Net with Steganalysis Detection**")

tab1, tab2, tab3 = st.tabs(["Encode", "Decode", "Detect"])

with tab1:
    st.header("Hide a Secret Image inside a Cover Image")
    col1, col2 = st.columns(2)
    with col1:
        cover_file = st.file_uploader("Upload Cover Image", type=['jpg','jpeg','png'], key='cover')
    with col2:
        secret_file = st.file_uploader("Upload Secret Image", type=['jpg','jpeg','png'], key='secret')

    if cover_file and secret_file:
        cover_img = Image.open(cover_file)
        secret_img = Image.open(secret_file)
        cover_tensor = preprocess(cover_img)
        secret_tensor = preprocess(secret_img)

        with st.spinner("Loading models and encoding..."):
            hidenet, _, _ = load_models()
            stego = hidenet.predict([cover_tensor, secret_tensor])

        stego_img = postprocess(stego)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(cover_img, caption="Cover Image", use_column_width=True)
        with col2:
            st.image(secret_img, caption="Secret Image", use_column_width=True)
        with col3:
            st.image(stego_img, caption="Stego Image", use_column_width=True)

        diff = np.abs(cover_tensor[0] - stego[0]) * 10
        diff_img = Image.fromarray((diff * 255).clip(0, 255).astype(np.uint8))
        st.image(diff_img, caption="Difference Map (x10)", width=300)

        from io import BytesIO
        buf = BytesIO()
        stego_img.save(buf, format='PNG')
        st.download_button("Download Stego Image", buf.getvalue(), "stego.png", "image/png")

with tab2:
    st.header("Reveal the Hidden Secret Image")
    stego_file = st.file_uploader("Upload Stego Image", type=['jpg','jpeg','png'], key='stego')
    if stego_file:
        stego_img = Image.open(stego_file)
        stego_tensor = preprocess(stego_img)

        with st.spinner("Decoding..."):
            _, revealnet, _ = load_models()
            revealed = revealnet.predict(stego_tensor)

        revealed_img = postprocess(revealed)
        col1, col2 = st.columns(2)
        with col1:
            st.image(stego_img, caption="Stego Image", use_column_width=True)
        with col2:
            st.image(revealed_img, caption="Revealed Secret", use_column_width=True)

with tab3:
    st.header("Detect Hidden Content")
    detect_file = st.file_uploader("Upload Image to Analyze", type=['jpg','jpeg','png'], key='detect')
    if detect_file:
        detect_img = Image.open(detect_file)
        detect_tensor = preprocess(detect_img)

        with st.spinner("Analyzing..."):
            _, _, detector = load_models()
            prob = detector.predict(detect_tensor)[0][0]

        st.image(detect_img, caption="Analyzed Image", width=300)
        st.metric("Steganography Probability", f"{prob*100:.1f}%")
        if prob > 0.5:
            st.error("Hidden content detected!")
        else:
            st.success("No hidden content detected.")