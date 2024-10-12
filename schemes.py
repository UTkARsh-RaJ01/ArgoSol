import streamlit as st

def display_content_with_links():
    """Display text and link content."""
    contents = [
        ("Paramparagat Krishi Vikas Yojana", "https://dmsouthwest.delhi.gov.in/scheme/paramparagat-krishi-vikas-yojana/"),
        ("Soil health Card scheme", "https://soilhealth.dac.gov.in/home"),
        ("PM Kisan", "https://pmkisan.gov.in/"),
        ("Pradhan Mantri Krishi Sinchai Yojana", "https://pmksy.gov.in/"),
        ("Pradhan Mantri Fasal Bima Yojana", "https://pmfby.gov.in/"),
        ("Kisan Credit Card", "https://www.myscheme.gov.in/schemes/kcc"),
        ("Modified Interest Subvention Scheme (MISS) provides concessional short-term agri-loans", "https://www.nabard.org/content1.aspx?id=602&catid=23&mid=23"),
        ("Jharkhand Krishi Rin Mafi Yojana", "https://jkrmy.jharkhand.gov.in/"),
        ("Rainfed Area Development (RAD)", "https://nmsa.dac.gov.in/frmComponents.aspx"),
        ("Digital Agriculture", "https://agriwelfare.gov.in/en/DigiAgriDiv"),
    ]
    
    content_html = "<div class='content'>"
    for text, link in contents:
        content_html += f"<p class='example-text'><strong>{text}</strong> -> <a href='{link}' class='example-link'>{link}</a></p>"
    content_html += "</div>"
    return content_html

st.markdown(
    """
    <style>
    .content {
        text-align: left;
        line-height: 1.6;
        margin: 20px;
    }
    .example-text {
        font-size: 20px; /* Font size for example text */
    }
    .example-link {
        color: lightblue;
    }
    h1 {
        font-size: 50px; /* Font size for heading */
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("<h1>Links</h1>", unsafe_allow_html=True)
st.markdown(display_content_with_links(), unsafe_allow_html=True)
