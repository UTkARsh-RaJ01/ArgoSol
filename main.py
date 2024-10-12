import streamlit as st


PAGES = {
    "Home": "home_page.py",
    "Reading": "sensor.py",
    
    "Disease": "crop_predict.py",
    "Crop": "cropdescription.py",
    "Erosion": "erosion.py",
    
   
    "Schemes": "schemes.py",
    "soil": "soil.py",
  
}

languages = {
    'English': 'en',
    'Hindi': 'hi',
    'Marathi': 'mr',
    'Punjabi': 'pa',
    'Bengali': 'bn',
    'Gujarati': 'gu'
}


if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'language_code' not in st.session_state:
    st.session_state.language_code = 'hi'


selected_language_name = st.sidebar.selectbox("Home Page Select language:", list(languages.keys()))
selected_language_code = languages[selected_language_name]
st.session_state.language_code = selected_language_code


st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()), index=list(PAGES.keys()).index(st.session_state.page))

st.session_state.page = selection


page_file = PAGES[st.session_state.page]


with open(page_file) as f:
    exec(f.read())
