# app.py
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import base64
from sklearn.preprocessing import LabelEncoder

# ----------------------
# Load model and fertilizer encoder
# ----------------------
model = pickle.load(open("classifier.pkl", "rb"))
ferti_encoder = pickle.load(open("fertilizer.pkl", "rb"))

# ----------------------
# Define categories manually
# ----------------------
soil_classes = ['Sandy', 'Loamy', 'Clayey', 'Red', 'Black']
crop_classes = ['rice', 'Wheat', 'Tobacco', 'Sugarcane', 'Pulses', 'pomegranate',
                'Paddy', 'Oil seeds', 'Millets', 'Maize', 'Ground Nuts', 'Cotton',
                'coffee', 'watermelon', 'Barley', 'kidneybeans', 'orange']

encode_soil = LabelEncoder()
encode_soil.fit(soil_classes)

encode_crop = LabelEncoder()
encode_crop.fit(crop_classes)

# ----------------------
# Fertilizer Info Dictionary
# ----------------------
fertilizer_info = {
    "Urea": {
        "en": {"NPK": "46-0-0", "Usage": "Boosts nitrogen levels; use during vegetative stage.",
               "Dosage": "50-70 kg per acre"},
        "hi": {"NPK": "46-0-0", "Usage": "नाइट्रोजन की मात्रा बढ़ाता है; वृद्धि अवस्था में प्रयोग करें।",
               "Dosage": "50-70 किग्रा प्रति एकड़"}
    },
    "DAP": {
        "en": {"NPK": "18-46-0", "Usage": "High phosphorus content; good for root development.",
               "Dosage": "40-50 kg per acre"},
        "hi": {"NPK": "18-46-0", "Usage": "उच्च फॉस्फोरस; जड़ों के विकास के लिए अच्छा।",
               "Dosage": "40-50 किग्रा प्रति एकड़"}
    },
    "Potash": {
        "en": {"NPK": "0-0-60", "Usage": "Increases disease resistance & improves grain filling.",
               "Dosage": "20-30 kg per acre"},
        "hi": {"NPK": "0-0-60", "Usage": "रोग प्रतिरोधक क्षमता बढ़ाता है और दाने भरने में सुधार करता है।",
               "Dosage": "20-30 किग्रा प्रति एकड़"}
    }
}

# ----------------------
# Add Background Image (works with base64)
# ----------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call function for background (put a file bg.jpg/png in same folder)
add_bg_from_local("bg.jpg")

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="KrishiSahay🌱", layout="centered")
st.title("🌱 KrishiSahay – Smart Fertilizer Recommender")

# Language toggle
lang = st.sidebar.radio("🌐 Language / भाषा", ("English", "हिंदी"))
lang_code = "en" if lang == "English" else "hi"

if lang_code == "en":
    st.write("Provide soil & crop details to get the best fertilizer suggestion.")
else:
    st.write("मिट्टी और फसल की जानकारी दें ताकि सर्वोत्तम उर्वरक सुझाव मिल सके।")

# Sidebar inputs
st.sidebar.header("Input Parameters / इनपुट पैरामीटर")

soil_type = st.sidebar.selectbox("🧑‍🌾 Soil Type / मिट्टी का प्रकार", soil_classes)
crop_type = st.sidebar.selectbox("🌿 Crop Type / फसल का प्रकार", crop_classes)

nitrogen = st.sidebar.number_input("🧪 Nitrogen (N)", min_value=0, max_value=200, value=20)
potassium = st.sidebar.number_input("🧪 Potassium (K)", min_value=0, max_value=200, value=30)
phosphorous = st.sidebar.number_input("🧪 Phosphorous (P)", min_value=0, max_value=200, value=40)
ph = st.sidebar.number_input("⚗️ Soil pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.sidebar.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

# Encode categorical inputs
soil_encoded = encode_soil.transform([soil_type])[0]
crop_encoded = encode_crop.transform([crop_type])[0]

# Average nutrient feature
avg_nutrients = (nitrogen + potassium + phosphorous) / 3

# Prepare input (8 features as expected by model)
features = np.array([[soil_encoded, crop_encoded, nitrogen, potassium, phosphorous, ph, rainfall, avg_nutrients]])

# Predict
if st.sidebar.button("🔍 Recommend Fertilizer / उर्वरक सुझाएं"):
    prediction = model.predict(features)[0]
    fertilizer = ferti_encoder.inverse_transform([prediction])[0]

    if lang_code == "en":
        st.success(f"✅ Recommended Fertilizer: **{fertilizer}**")
    else:
        st.success(f"✅ अनुशंसित उर्वरक: **{fertilizer}**")

    # Show Fertilizer Details
    if fertilizer in fertilizer_info:
        st.subheader("📋 Fertilizer Details / उर्वरक विवरण")
        st.write(f"**NPK Ratio:** {fertilizer_info[fertilizer][lang_code]['NPK']}")
        st.write(f"**Usage:** {fertilizer_info[fertilizer][lang_code]['Usage']}")
        st.write(f"**Recommended Dosage:** {fertilizer_info[fertilizer][lang_code]['Dosage']}")

    # Nutrient balance chart
    st.subheader("📊 Soil Nutrient Balance / मिट्टी पोषक संतुलन")
    nutrients = ["Nitrogen (N)", "Phosphorous (P)", "Potassium (K)"]
    values = [nitrogen, phosphorous, potassium]

    fig, ax = plt.subplots()
    ax.bar(nutrients, values, color=['green', 'orange', 'blue'])
    ax.set_ylabel("Level")
    ax.set_title("Soil Nutrient Balance")
    st.pyplot(fig)

    # Alternative top-3 fertilizers
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        top_indices = np.argsort(probs)[::-1][:3]

        if lang_code == "en":
            st.subheader("🔄 Top Fertilizer Alternatives")
        else:
            st.subheader("🔄 शीर्ष वैकल्पिक उर्वरक")

        for idx in top_indices:
            fert_name = ferti_encoder.inverse_transform([idx])[0]
            if lang_code == "en":
                st.write(f"- {fert_name} ({probs[idx] * 100:.2f}%)")
            else:
                st.write(f"- {fert_name} ({probs[idx] * 100:.2f}%)")
# app.py (only background + transparency part updated)

# Transparent background style
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("bg.jpg");
    background-size: cover;
    background-position: center;
}}

.block-container {{
    background: rgba(255, 255, 255, 0.7);  /* semi-transparent white */
    border-radius: 15px;
    padding: 20px;
}}

[data-testid="stSidebar"] > div:first-child {{
    background: rgba(255, 255, 255, 0.7);
    border-radius: 12px;
    padding: 15px;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
