import streamlit as st
import pandas as pd
import joblib
from googletrans import Translator

model = joblib.load('crop_recommendation_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
imputer = joblib.load('imputer.pkl')

translator = Translator()


languages = {
    'English': 'en',
    'Hindi': 'hi',
    'Tamil': 'ta',
    'Gujarati': 'gu',
}

def translate_text(text, dest_language):
    translated = translator.translate(text, dest=dest_language)
    return translated.text


selected_language = st.sidebar.selectbox("Select Language", list(languages.keys()))

st.title(translate_text('Crop Recommendation System', languages[selected_language]))

nitrogen = st.number_input(translate_text('Nitrogen', languages[selected_language]), min_value=0.0, step=0.1)
phosphorous = st.number_input(translate_text('Phosphorous', languages[selected_language]), min_value=0.0, step=0.1)
potassium = st.number_input(translate_text('Potassium', languages[selected_language]), min_value=0.0, step=0.1)
temperature = st.number_input(translate_text('Temperature', languages[selected_language]), min_value=0.0, step=0.1)
humidity = st.number_input(translate_text('Humidity', languages[selected_language]), min_value=0.0, step=0.1)
ph_value = st.number_input(translate_text('pH Value', languages[selected_language]), min_value=0.0, step=0.1)
rainfall = st.number_input(translate_text('Rainfall', languages[selected_language]), min_value=0.0, step=0.1)


input_data = pd.DataFrame({
    'Nitrogen': [nitrogen],
    'Phosphorus': [phosphorous], 
    'Potassium': [potassium],
    'Temperature': [temperature],
    'Humidity': [humidity],
    'pH_Value': [ph_value],  
    'Rainfall': [rainfall]
})

input_data.columns = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']


if (input_data == 0.0).all().all():
    st.write(translate_text('No Crop', languages[selected_language]))
else:
    input_data = imputer.transform(input_data)
    prediction = model.predict(input_data)
    predicted_crop = label_encoder.inverse_transform(prediction)
    st.write(f'{translate_text("Predicted Crop:", languages[selected_language])} {predicted_crop[0]}')
