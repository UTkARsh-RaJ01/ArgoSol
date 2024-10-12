import streamlit as st
from googletrans import Translator
from PIL import Image

translator = Translator()

def translate_text(text, target_language='hi'):
    """Translate text into the specified target language."""
    result = translator.translate(text, dest=target_language)
    return result.text

def translate_page(elements, target_language='hi'):
    """Translate a list of elements into the specified target language."""
    translated_elements = [translate_text(element, target_language) for element in elements]
    return translated_elements

def display_images(target_language='hi'):
    """Display images with translated captions and navigation buttons."""
    st.write("### Important Links")
    image_info = [
        ("Crop Production Method", "Crop Production.png", "Crop"),
        ("Disease", "Plant Infection.png", "Disease"),
        ("Government Schemes", "Government Schemes.png", "Schemes"),
        ("Soil Health", "Soil Health.png", "soil"),
        ("Erosion", "erosion.jpg", "Erosion"),
        ("Reading", "tech.jpg", "Reading"),
        
    ]

    image_titles = [item[0] for item in image_info]
    translated_titles = translate_page(image_titles, target_language)

    num_cols = 4
    cols = st.columns(num_cols)

    target_size = (200, 200)  

    for i, (original_title, img, page) in enumerate(image_info):
        with cols[i % num_cols]:
            image = Image.open(img)
            resized_image = image.resize(target_size) 
            st.image(resized_image, caption=translated_titles[i], use_column_width=True)
            if page:
                if st.button(translated_titles[i]):
                    st.session_state.page = page

def home_page(target_language='hi'):
    """Display the home page with translated content."""
    st.title(translate_text("Home Page", target_language))
    st.write(translate_text("Welcome to the Home page!", target_language))
    display_images(target_language)

if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'language_code' not in st.session_state:
    st.session_state.language_code = 'hi'

home_page(st.session_state.language_code)




