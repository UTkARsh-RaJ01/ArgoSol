import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from googletrans import Translator

try:
    model = tf.keras.models.load_model('my_model.h5')
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

class_names = ['Black Soil', 'Cinder Soil', 'Laterite Soil', 'Peat Soil', 'Yellow Soil']

crop_recommendations = {
    'Black Soil': ('Wheat', 'wheat.jpg'),
    'Cinder Soil': ('Barley', 'barley.jpg'),
    'Laterite Soil': ('Cassava', 'cassava.jpg'),
    'Peat Soil': ('Rice', 'rice.webp'),
    'Yellow Soil': ('Maize', 'maize.webp')
}

translator = Translator()

def translate_text(text, target_language='hi'):
    result = translator.translate(text, dest=target_language)
    return result.text

def predict_image(image):
    img = image.resize((220, 220))  
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    soil_type = class_names[predicted_class[0]]
    
    crop, crop_image_path = crop_recommendations[soil_type]
    crop_image = Image.open(crop_image_path)
    
    return soil_type, crop, crop_image

if 'language' not in st.session_state:
    st.session_state.language = 'en'

def set_language(lang):
    st.session_state.language = lang

language_names = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "pa": "Punjabi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "ta": "Tamil"
}

language_codes = list(language_names.keys())
language_names_list = list(language_names.values())

st.sidebar.title("Language Selection")
target_language = st.sidebar.selectbox(
    "Select language:", language_names_list, index=language_names_list.index(language_names[st.session_state.language])
)


target_language_code = [code for code, name in language_names.items() if name == target_language][0]
set_language(target_language_code)

st.title(translate_text("Soil Type Prediction and Crop Recommendation", st.session_state.language))

uploaded_file = st.file_uploader(translate_text("Choose an image...", st.session_state.language), type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=translate_text('Uploaded Image', st.session_state.language), use_column_width=True)
    
    if st.button(translate_text('Predict', st.session_state.language)):
        with st.spinner(translate_text('Predicting...', st.session_state.language)):
            try:
                soil_type, crop, crop_image = predict_image(image)
                st.write(translate_text(f"Predicted Soil Type: {soil_type}", st.session_state.language))
                st.write(translate_text(f"Recommended Crop: {crop}", st.session_state.language))
                st.image(crop_image, caption=translate_text(f"Recommended Crop: {crop}", st.session_state.language), use_column_width=True)
            except Exception as e:
                st.error(translate_text(f"Error during prediction: {e}", st.session_state.language))
