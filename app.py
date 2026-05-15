import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="SkinTell",
    page_icon="🧴",
    layout="centered"
)

# ── Constants ────────────────────────────────────────────────────
CLASS_NAMES = ['Clear', 'Mild', 'Moderate', 'Severe']
COLORS      = ['#9ccfd8', '#f6c177', '#e07b7b', '#c4a7e7']
IMG_SIZE    = (224, 224)

DESCRIPTIONS = {
    'Clear':    'No significant acne detected. Skin appears healthy.',
    'Mild':     'Minor breakouts present. A gentle skincare routine may help.',
    'Moderate': 'Moderate acne detected. Consider consulting a dermatologist.',
    'Severe':   'Severe acne detected. A dermatologist visit is recommended.'
}

# ── Load model ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/best_model.keras')

# ── UI ───────────────────────────────────────────────────────────
st.title("🧴 SkinTell")
st.subheader("Acne Severity Classifier")
st.write("Upload a clear photo of skin and SkinTell will classify the acne severity level.")

st.divider()

uploaded_file = st.file_uploader("Upload a skin photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    # Preprocess
    img_resized = image.resize(IMG_SIZE)
    img_array  = np.array(img_resized) / 255.0
    img_array  = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Analyzing..."):
        model = load_model()
        predictions = model.predict(img_array)[0]

    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence      = np.max(predictions)
    color           = COLORS[np.argmax(predictions)]

    with col2:
        st.markdown(f"### Result")
        st.markdown(
            f"<div style='background-color:{color};padding:20px;border-radius:12px;text-align:center'>"
            f"<h2 style='color:white;margin:0'>{predicted_class}</h2>"
            f"<p style='color:white;margin:4px 0 0 0'>{confidence:.0%} confidence</p>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.write("")
        st.info(DESCRIPTIONS[predicted_class])

    # Probability bar chart
    st.divider()
    st.markdown("#### Confidence per class")
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(CLASS_NAMES, predictions, color=COLORS, edgecolor='white')
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    for bar, val in zip(bars, predictions):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.0%}', va='center', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

st.divider()
st.caption("⚠️ SkinTell is a learning project and is not a medical diagnostic tool. Always consult a dermatologist for medical advice.")
