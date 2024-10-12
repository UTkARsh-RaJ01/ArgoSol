import streamlit as st
from googletrans import Translator

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
dest_language = languages[selected_language]


st.title(translate_text('10 Detailed Ways to Prevent Soil Erosion', dest_language))


st.write(translate_text("""
Soil erosion is a critical environmental challenge that can lead to the loss of fertile topsoil, reduced agricultural productivity, and increased sedimentation in water bodies. Understanding and implementing effective soil erosion prevention methods is crucial for maintaining soil health and preventing land degradation. Here are 10 detailed methods to prevent soil erosion:
""", dest_language))


methods = {
    "1. Contour Plowing": """
        Contour plowing involves plowing along the natural contours of the land rather than up and down slopes. This technique helps to slow down the flow of water, allowing it to be absorbed into the soil. By creating furrows that run perpendicular to the slope, contour plowing reduces runoff and helps to capture and retain soil. This method can also improve soil fertility and reduce the likelihood of water erosion.
    """,
    "2. Terracing": """
        Terracing is the practice of creating stepped levels on steep slopes to reduce water runoff and soil erosion. By dividing a slope into a series of flat or gently sloping terraces, this technique slows down the flow of water and captures sediment. Terracing not only prevents erosion but also makes it possible to cultivate crops on steep terrain, turning previously unusable land into productive agricultural areas.
    """,
    "3. Cover Crops": """
        Cover crops are plants grown specifically to cover the soil during periods when it would otherwise be bare. Common cover crops include legumes, grasses, and other plants that protect the soil from wind and water erosion. These plants improve soil structure, increase organic matter, and enhance soil fertility by fixing nitrogen and preventing erosion. Cover crops also help to reduce soil compaction and increase water infiltration.
    """,
    "4. Reforestation": """
        Reforestation involves planting trees and restoring forested areas to stabilize soil and prevent erosion. Trees have extensive root systems that bind soil together, reduce water runoff, and absorb excess nutrients and pollutants. Reforestation helps to restore natural habitats, improve biodiversity, and mitigate the impacts of climate change by capturing carbon dioxide and enhancing the soil’s ability to retain moisture.
    """,
    "5. Mulching": """
        Mulching involves applying a layer of material, such as organic matter (e.g., straw, leaves) or inorganic materials (e.g., gravel, plastic), to the soil surface. Mulch protects the soil from erosion by reducing the impact of rainfall, preventing water runoff, and maintaining soil moisture. It also helps to suppress weed growth, improve soil fertility, and enhance the overall health of the soil.
    """,
    "6. Gully Plugging": """
        Gully plugging involves installing barriers or structures in gullies to slow water flow and reduce erosion in these areas. By creating physical barriers, such as check dams or silt fences, gully plugging helps to capture sediment, stabilize the gully walls, and prevent further erosion. This technique is particularly useful for managing erosion in areas with concentrated water flow and steep slopes.
    """,
    "7. Grass Waterways": """
        Grass waterways are vegetated channels that help to manage water flow and prevent soil erosion in natural or constructed drainage paths. By planting grass or other vegetation in these channels, the water flow is slowed, and the soil is stabilized. Grass waterways help to reduce the velocity of surface runoff, capture sediment, and prevent gully formation in areas prone to erosion.
    """,
    "8. Windbreaks": """
        Windbreaks are rows of trees or shrubs planted to act as barriers against wind. By reducing wind speed, windbreaks minimize soil erosion caused by wind. They also help to protect crops, reduce moisture evaporation, and create a more favorable microclimate for plant growth. Windbreaks can be designed to protect fields, orchards, and other agricultural areas from the damaging effects of strong winds.
    """,
    "9. Riparian Buffers": """
        Riparian buffers are vegetated areas located along waterways that help to filter runoff and protect streambanks from erosion. By establishing native vegetation in these buffer zones, riparian buffers capture and filter pollutants, reduce sedimentation, and stabilize streambanks. They also provide habitat for wildlife, enhance water quality, and improve the overall health of aquatic ecosystems.
    """,
    "10. Erosion Control Fabrics": """
        Erosion control fabrics, also known as erosion control mats or blankets, are used to stabilize soil and prevent erosion on disturbed or bare soil. These fabrics, made from materials such as jute, coir, or synthetic fibers, provide a protective cover that reduces soil erosion caused by wind and water. They also help to retain soil moisture, promote seed germination, and support vegetation establishment.
    """
}


for method, details in methods.items():
    translated_method = translate_text(method, dest_language)
    translated_details = translate_text(details, dest_language)
    st.write(translated_method)
    st.write(translated_details)

st.write(translate_text("Implementing these methods can significantly reduce soil erosion and promote sustainable land management.", dest_language))
