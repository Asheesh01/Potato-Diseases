# -------------------- IMPORTS --------------------
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
from gtts import gTTS
from deep_translator import GoogleTranslator
import tempfile
import tensorflow as tf
from streamlit_lottie import st_lottie
import requests
import random
from streamlit_extras.let_it_rain import rain

# Disable TensorFlow logs
tf.get_logger().setLevel('ERROR')

# -------------------- CONFIG --------------------
st.set_page_config(page_title="🥔 Potato Disease Detector", layout="centered", page_icon="🥔")

# -------------------- STYLE --------------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #1b2735, #090a0f);
}
h1, h2, h3, h4, h5, h6, p, div {
    color: white;
}
hr {border: 1px solid #444;}
button[kind="secondary"] {
    background-color: #00b894 !important;
    color: white !important;
    border-radius: 12px !important;
    font-weight: bold !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
model = load_model('potato_vgg16.h5')
labels = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

if not os.path.exists("predictions.csv"):
    pd.DataFrame(columns=["Image", "Prediction", "Confidence", "Time", "Image_Path"]).to_csv("predictions.csv", index=False)

# -------------------- HELPER FUNCTIONS --------------------
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def translate_text(text, target_lang="hi"):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except:
        return text

def is_potato_leaf(img_array):
    """Heuristic to check if uploaded image looks like a potato leaf"""
    img_array = np.array(img_array).astype('float32') / 255.0
    avg_r, avg_g, avg_b = np.mean(img_array[:, :, 0]), np.mean(img_array[:, :, 1]), np.mean(img_array[:, :, 2])
    green_ratio = avg_g / (avg_r + avg_b + 1e-5)
    brightness = np.mean(img_array)
    return not (green_ratio > 1.4 or brightness > 0.8 or brightness < 0.1 or green_ratio < 0.5)

def play_voice_feedback(text, lang):
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            st.audio(tmp_file.name, format="audio/mp3")
    except:
        st.error("❌ Voice feedback failed.")

# -------------------- DISEASE DATA --------------------
disease_info = {
    'Potato___Early_blight': {
        'name': 'Early Blight',
        'cause': 'Fungus Alternaria solani',
        'symptoms': "Dark concentric spots on older leaves forming target-like rings.",
        'advice': "- Remove infected leaves and avoid overhead irrigation.\n- Apply fungicides (chlorothalonil, copper-based).\n- Maintain proper spacing and crop rotation."
    },
    'Potato___Late_blight': {
        'name': 'Late Blight',
        'cause': 'Pathogen Phytophthora infestans',
        'symptoms': "Water-soaked lesions turning brown to black; white mold may appear under humid conditions.",
        'advice': "- Immediately remove infected plants.\n- Apply fungicides like mancozeb or metalaxyl.\n- Avoid wetting leaves during irrigation.\n- Use resistant varieties if available."
    },
    'Potato___healthy': {
        'name': 'Healthy Leaf',
        'cause': 'No disease detected',
        'symptoms': "Leaf appears green and uniform with no visible spots.",
        'advice': "- Maintain good soil health.\n- Ensure balanced fertilization and irrigation.\n- Continue regular monitoring and preventive sprays."
    }
}

# -------------------- SESSION STATE --------------------
if "page" not in st.session_state:
    st.session_state.page = "👋 Welcome"

# -------------------- SIDEBAR --------------------
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to", ["👋 Welcome", "🏠 Home", "📜 History", "ℹ️ About"],
    index=["👋 Welcome", "🏠 Home", "📜 History", "ℹ️ About"].index(st.session_state.page)
)
st.session_state.page = page

st.sidebar.markdown("### 🌐 Language Options")
language = st.sidebar.selectbox("Select Language", ["English", "Hindi"])
voice_feedback = st.sidebar.checkbox("🔊 Play Voice Feedback", value=False)

# =====================================================
#                   WELCOME PAGE
# =====================================================
if st.session_state.page == "👋 Welcome":

    # --- Fade-In Effect ---
    st.markdown("""
    <style>
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    div[data-testid="stAppViewContainer"] {
        animation: fadeIn 1.5s ease-in-out;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Particle Background ---
    st.components.v1.html("""
    <div id="particles-js"></div>
    <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
    <script>
    particlesJS("particles-js", {
      "particles": {
        "number": {"value": 75},
        "color": {"value": "#00ffcc"},
        "shape": {"type": "circle"},
        "opacity": {"value": 0.5},
        "size": {"value": 3},
        "line_linked": {"enable": true, "color": "#00ffff"},
        "move": {"enable": true, "speed": 1.5}
      },
      "interactivity": {"events": {"onhover": {"enable": true, "mode": "grab"}}}
    });
    </script>
    <style>
    #particles-js {
      position: fixed;
      width: 100%;
      height: 100%;
      z-index: -1;
      top: 0;
      left: 0;
    }
    </style>
    """, height=600)

    # --- Title and Subtitle ---
    st.markdown("""
    <h1 style='text-align:center; color:#00ffcc;'>
        🥔 Potato Leaf Disease Detection System
    </h1>
    <h3 style='text-align:center; color:#FFD700;'>
        AI-Powered Crop Health Assistant 🌿
    </h3>
    <p style='text-align:center; color:#ccc;'>
        Upload or capture a potato leaf image to detect diseases instantly.<br>
        Empowering Smart Farming through Artificial Intelligence.
    </p>
    """, unsafe_allow_html=True)

    # --- Lottie Animation ---
    ai_anim = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_ygiuluqn.json")
    st_lottie(ai_anim, height=280, key="ai_anim")

    # --- Falling Leaf Emoji Animation ---
    rain(emoji="🍃", font_size=25, falling_speed=5, animation_length="infinite")

    # --- Start Detection Button ---
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Start Detection", use_container_width=True):
        st.session_state.page = "🏠 Home"
        st.rerun()

    # --- Potato Facts ---
    potato_facts = [
        "🥔 Potatoes were the first vegetable grown in space!",
        "🌱 One potato plant can produce up to 10 potatoes.",
        "💧 Avoid watering potato leaves at night to prevent blight.",
        "🌿 Late blight can destroy an entire field within days!",
        "🍀 Proper crop rotation prevents fungal infections."
    ]
    st.info(random.choice(potato_facts))

    st.markdown("""
    <hr>
    <p style='text-align:center; color:gray;'>
        🌾 Built with <b>TensorFlow</b> + <b>Streamlit</b> • Modern AI for Smart Farming
    </p>
    """, unsafe_allow_html=True)

# =====================================================
#                   HOME PAGE
# =====================================================
elif st.session_state.page == "🏠 Home":
    st.title("🥔 Potato Leaf Disease Detection")
    st.write("Upload or capture a potato leaf image to get a diagnosis and treatment advice.")

    option = st.radio("Select Image Source:", ["📂 Upload from Device", "📸 Capture from Camera"])
    uploaded = st.file_uploader("📷 Upload Image", type=['jpg', 'png', 'jpeg']) if option == "📂 Upload from Device" else st.camera_input("📸 Capture Photo")

    if uploaded:
        st.image(uploaded, caption="Uploaded Image", width="stretch")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp.write(uploaded.getbuffer())
            img_path = temp.name

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = np.array(img)

        if not is_potato_leaf(img_array):
            st.error("🚫 This image doesn’t appear to be a potato leaf. Please upload a valid potato leaf photo.")
            st.stop()

        img_input = np.expand_dims(img_array / 255.0, axis=0)
        pred = model.predict(img_input)
        label = labels[np.argmax(pred)]
        confidence = round(np.max(pred) * 100, 2)
        info = disease_info[label]
        if language == "Hindi":
            info = {k: translate_text(v, "hi") for k, v in info.items()}

        st.markdown(f"### 🩺 Prediction: **{info['name']}**")
        st.markdown(f"**Confidence:** {confidence}%")
        st.markdown(f"**Cause:** {info['cause']}")
        st.markdown(f"**Symptoms:** {info['symptoms']}")
        st.info(info['advice'])

        fig, ax = plt.subplots()
        ax.bar(labels, pred[0], color=['#ff9999', '#66b3ff', '#99ff99'])
        ax.set_ylabel('Confidence')
        ax.set_ylim([0, 1])
        plt.xticks(rotation=15)
        st.pyplot(fig)

        if label == 'Potato___healthy':
            st.success("✅ Your plant is healthy!")
        else:
            st.warning("⚠️ Disease detected! Follow the advice above.")

        if voice_feedback:
            text = f"The result is {info['name']}. {info['advice']}" if language == "English" else f"परिणाम है {info['name']}। {info['advice']}"
            play_voice_feedback(text, 'hi' if language == "Hindi" else 'en')

        os.makedirs("history_images", exist_ok=True)
        save_path = os.path.join("history_images", uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        pd.DataFrame([[uploaded.name, info['name'], confidence, datetime.datetime.now(), save_path]],
                     columns=["Image", "Prediction", "Confidence", "Time", "Image_Path"]).to_csv("predictions.csv", mode='a', header=False, index=False)
        st.success("🗂️ Prediction and image saved to history.")
    else:
        st.info("Please upload or capture an image to start prediction.")

# =====================================================
#                   HISTORY PAGE
# =====================================================
elif st.session_state.page == "📜 History":
    st.title("📜 Prediction History")
    if os.path.exists("predictions.csv"):
        df = pd.read_csv("predictions.csv", on_bad_lines='skip')
        if not df.empty:
            for _, row in df.tail(20).iloc[::-1].iterrows():
                with st.expander(f"🩺 {row['Prediction']} ({row['Confidence']}%) - {row['Time']}"):
                    if os.path.exists(row["Image_Path"]):
                        st.image(row["Image_Path"], caption=row["Image"], width="stretch")
                    st.markdown(f"**Prediction:** {row['Prediction']} ({row['Confidence']}%)")
            if st.button("🗑️ Clear History"):
                os.remove("predictions.csv")
                pd.DataFrame(columns=["Image", "Prediction", "Confidence", "Time", "Image_Path"]).to_csv("predictions.csv", index=False)
                st.success("✅ History cleared successfully.")
        else:
            st.info("No prediction history yet.")
    else:
        st.warning("No history file found.")

# =====================================================
#                   ABOUT PAGE
# =====================================================
elif st.session_state.page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ### 🥔 Potato Leaf Disease Detection System
    - **Developed by:** Shruti & Asheesh  
    - **Dataset Source:** [PlantVillage - Potato Leaf Dataset (Kaggle)](https://www.kaggle.com/datasets/aarishasifkhan/plantvillage-potato-disease-dataset)
    - **Model Used:** VGG16 (Transfer Learning)
    - **Framework:** TensorFlow + Streamlit  
    - **Accuracy:** ~95%
    """)

# =====================================================
#                   FOOTER
# =====================================================
st.markdown("""
<hr style='border:1px solid #555;'>
<p style='text-align:center; color:gray;'>
© 2025 Potato Disease Detection | Built with ❤️ using Streamlit
</p>
""", unsafe_allow_html=True)
