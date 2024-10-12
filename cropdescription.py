import streamlit as st
import pandas as pd
from googletrans import Translator
from PIL import Image
import os


file_path = 'crop.xlsx'
df = pd.read_excel(file_path)


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


st.title(translate_text("Crop Information", languages[selected_language]))
crop_select_label = translate_text("Select a crop:", languages[selected_language])
info_header = translate_text("Information about", languages[selected_language])
method_label = translate_text("Method of Growing", languages[selected_language])
soil_label = translate_text("Soil Required", languages[selected_language])
disease_label = translate_text("Common Diseases", languages[selected_language])
season_label = translate_text("Season", languages[selected_language])
description_label = translate_text("Description", languages[selected_language])

crop_name = st.selectbox(crop_select_label, df['Crop'].unique())


if crop_name:
    crop_info = df[df['Crop'] == crop_name].iloc[0]
    
  
    method_of_growing = translate_text(crop_info['Method of Growing'], languages[selected_language])
    soil_required = translate_text(crop_info['Soil Required'], languages[selected_language])
    common_diseases = translate_text(crop_info['Common Diseases'], languages[selected_language])
    season = translate_text(crop_info['Season'], languages[selected_language])
    description = translate_text(crop_info['Description'], languages[selected_language])
    
    st.subheader(f"{info_header} {crop_name}")
    st.write(f"**{method_label}:** {method_of_growing}")
    st.write(f"**{soil_label}:** {soil_required}")
    st.write(f"**{disease_label}:** {common_diseases}")
    st.write(f"**{season_label}:** {season}")
    st.write(f"**{description_label}:** {description}")



    image_path = os.path.join('Images', f"{crop_name}.jpg")  
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption=crop_name, use_column_width=True)
    else:
        st.write(translate_text("Image not available for this crop.", languages[selected_language]))