import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Dropout
from PIL import Image
from googletrans import Translator

translator = Translator()

def translate_text(text, dest_language):
    if text is None or text.strip() == "":
        return text
    translated = translator.translate(text, dest=dest_language)
    return translated.text

def create_model():
    conv = DenseNet121(weights='imagenet', include_top=False, input_shape=(256, 256, 3), pooling='avg')
    model = Sequential()
    model.add(conv)
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(38, activation='softmax')) 
    return model

model = create_model()
model.build(input_shape=(None, 256, 256, 3))


model.load_weights('my_plant_model1.weights.h5')


class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)_Powdery_mildew', 'Cherry_(including_sour)_healthy',
    'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)_Common_rust', 
    'Corn_(maize)_Northern_Leaf_Blight', 'Corn_(maize)_healthy', 'Grape___Black_rot',
    'Grape___Esca(Black_Measles)', 'Grape___Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

disease_treatments = {

    'Apple___Apple_scab': {
    'treatment': (
        "To manage Apple Scab, begin by removing and destroying fallen leaves in autumn to minimize the overwintering spores. Apply fungicides like captan or myclobutanil starting at the green tip stage and repeat applications every 7-10 days as necessary to maintain control."
    ),
    'suggested_changes': (
        "Ensure proper spacing of apple trees to improve air circulation, which helps in reducing humidity levels that favor scab development. Additionally, using resistant apple varieties can significantly reduce disease incidence."
    ),
    'errors_and_corrections': (
        "A common error is inadequate removal of fallen leaves, which can harbor spores. Ensure thorough leaf removal and dispose of them away from the orchard. Inconsistent fungicide application or inadequate coverage can also lead to ineffective disease control; ensure even and timely application."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},

'Apple___Black_rot': {
    'treatment': (
        "Prune out infected branches and remove mummified fruits to reduce the source of the disease. Apply fungicides such as thiophanate-methyl or captan starting at petal fall, with follow-up treatments every 10-14 days as needed."
    ),
    'suggested_changes': (
        "Improve orchard sanitation by removing infected plant material and maintaining proper tree pruning to enhance airflow and reduce humidity around the fruit."
    ),
    'errors_and_corrections': (
        "Failure to remove infected plant material can result in continued spread. Ensure thorough inspection and removal of diseased parts. Inconsistent fungicide application or inadequate coverage can also lead to ineffective disease control; ensure a regular schedule is maintained."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},

'Apple___Cedar_apple_rust': {
    'treatment': (
        "Remove galls from cedar trees during winter to reduce the spore source. Apply fungicides such as myclobutanil or mancozeb to apple trees starting at the pink bud stage, with applications every 7-10 days until 1-2 weeks after petal fall."
    ),
    'suggested_changes': (
        "Regularly monitor cedar trees near apple orchards and remove any galls to reduce the risk of spore dispersal. Consider planting apple varieties that are less susceptible to cedar apple rust."
    ),
    'errors_and_corrections': (
        "Overlooking cedar galls can lead to repeated infections. Ensure thorough monitoring and removal. Inadequate fungicide application may also lead to ineffective control; apply consistently and according to the recommended schedule."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},
'Apple___healthy': {
    'treatment': (
        "The apple trees are healthy when they show no symptoms of disease and are well-maintained with proper irrigation, fertilization, and pest management practices."
    ),
    'suggested_changes': None,
    'errors_and_corrections': None,
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},

'Blueberry___healthy': {
    'treatment': (
        "The blueberry plants are healthy when they exhibit no signs of disease and are receiving proper care in terms of irrigation, soil nutrition, and pest management."
    ),
    'suggested_changes': None,
    'errors_and_corrections': None,
    'environmental_conditions': (
        "pH: 4.5 - 5.5 | Nitrogen: 40 - 80 kg/ha | Phosphorus: 20 - 40 kg/ha | Potassium: 80 - 120 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},

'Cherry_(including_sour)_Powdery_mildew': {
    'treatment': (
        "Apply fungicides such as sulfur or myclobutanil when the disease is first detected. Repeat applications every 10-14 days as necessary. Ensure good air circulation and avoid overhead watering."
    ),
    'suggested_changes': (
        "Maintain proper spacing between cherry trees to enhance air flow and reduce humidity, which helps prevent the development of powdery mildew."
    ),
    'errors_and_corrections': (
        "Insufficient air circulation or excessive moisture from overhead watering can exacerbate the disease. Ensure proper spacing and use drip irrigation instead of overhead watering. Inconsistent fungicide application can also lead to increased disease pressure; follow the recommended schedule closely."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},

'Corn_(maize)_Cercospora_leaf_spot_Gray_leaf_spot': {
    'treatment': (
        "Rotate crops with non-host species to reduce disease carryover. Apply fungicides like azoxystrobin or pyraclostrobin at the first sign of disease and repeat every 14 days as needed."
    ),
    'suggested_changes': (
        "Implement a crop rotation plan that includes non-host crops to minimize the risk of disease persistence. Maintain optimal plant health through proper fertilization and irrigation."
    ),
    'errors_and_corrections': (
        "Failure to rotate crops or maintain soil health can lead to persistent disease problems. Ensure crop rotation and proper nutrient management. Inconsistent fungicide use may also contribute to disease; adhere to application guidelines and schedules."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},

'Corn_(maize)_Common_rust': {
    'treatment': (
        "Plant resistant hybrid varieties to reduce susceptibility. Apply fungicides such as propiconazole or azoxystrobin at the onset of symptoms, with follow-up treatments every 7-10 days as necessary."
    ),
    'suggested_changes': (
        "Utilize resistant corn varieties and practice good field management, including proper planting density and timely fertilization."
    ),
    'errors_and_corrections': (
        "Planting susceptible varieties or inadequate field management can increase disease pressure. Use resistant hybrids and maintain good field practices. Inconsistent fungicide application may also lead to poor disease control; ensure timely and thorough application."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},
'Corn_(maize)_Northern_Leaf_Blight': {
    'treatment': (
        "Remove and destroy infected plant residues from the previous season. Apply fungicides like tebuconazole or chlorothalonil when the disease is first noticed and repeat every 10-14 days."
    ),
    'suggested_changes': (
        "Implement crop rotation and avoid planting corn in fields where the disease was present in the previous season. Ensure proper plant spacing and manage irrigation to reduce leaf wetness."
    ),
    'errors_and_corrections': (
        "Failure to remove plant residues or improper fungicide application can contribute to disease persistence. Ensure residue management and follow recommended fungicide schedules. Inadequate spacing or irrigation management can also exacerbate the problem."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},
'Corn_(maize)_healthy': {
    'treatment': (
        "The corn plants are healthy when they are free from disease symptoms and are well-maintained with appropriate care, including irrigation, fertilization, and pest management practices."
    ),
    'suggested_changes': "",
    'errors_and_corrections': "",
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},
'Grape___Black_rot': {
    'treatment': (
        "Remove and destroy infected plant parts. Apply fungicides like myclobutanil or boscalid during the growing season, with applications every 7-10 days as needed."
    ),
    'suggested_changes': (
        "Ensure proper pruning and canopy management to improve air circulation and reduce humidity around the vines."
    ),
    'errors_and_corrections': (
        "Failure to remove infected plant parts or inconsistent fungicide application can lead to continued disease pressure. Regularly inspect and remove diseased parts and adhere to the recommended fungicide schedule."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},
'Grape___Esca(Black_Measles)': {
    'treatment': (
        "Remove and destroy infected wood and clusters. Apply fungicides like copper hydroxide or mancozeb at bud break and during the growing season, repeating every 10-14 days."
    ),
    'suggested_changes': (
        "Ensure proper vine management and pruning to improve air circulation and reduce humidity around the vines."
    ),
    'errors_and_corrections': (
        "Inadequate removal of infected wood or improper fungicide application can contribute to the disease. Ensure thorough sanitation and follow recommended application schedules for effective control."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},
'Grape___Leaf_blight(Isariopsis_Leaf_Spot)': {
    'treatment': (
        "Prune infected leaves and apply fungicides like copper sulfate or mancozeb at the onset of symptoms, repeating every 10-14 days."
    ),
    'suggested_changes': (
        "Improve vine canopy management and ensure proper spacing to enhance airflow and reduce humidity."
    ),
    'errors_and_corrections': (
        "Failure to manage vine canopy or inadequate fungicide coverage can lead to persistent disease issues. Maintain good vine management practices and ensure even fungicide application."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},
'Grape___healthy': {
    'treatment': (
        "The grape vines are healthy when they show no symptoms of disease and are well-maintained with proper irrigation, fertilization, and pest management practices."
    ),
    'suggested_changes': "",
    'errors_and_corrections': "",
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},
'Orange___Haunglongbing(Citrus_greening)': {
    'treatment': (
        "Remove and destroy infected trees and debris to reduce the spread of the disease. Apply insecticides to control the vector (Asian citrus psyllid) and use bactericides like streptomycin if recommended by local extension services."
    ),
    'suggested_changes': (
        "Implement strict vector control measures and avoid planting in areas with known vector populations. Regularly inspect trees and use certified disease-free planting material."
    ),
    'errors_and_corrections': (
        "Failure to control the vector or inadequate removal of infected trees can contribute to disease spread. Ensure vector control and removal of infected trees are conducted as recommended. Inconsistent application of bactericides can also lead to ineffective control; follow local guidelines carefully."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},

'Peach___Bacterial_spot': {
    'treatment': (
        "Remove and destroy infected plant material and apply copper-based bactericides or antibiotics like streptomycin during the growing season. Follow application schedules as recommended for effective control."
    ),
    'suggested_changes': (
        "Improve air circulation around peach trees by proper pruning and spacing to reduce humidity. Use resistant peach varieties if available."
    ),
    'errors_and_corrections': (
        "Inadequate removal of infected material or inconsistent bactericide application can lead to continued disease pressure. Ensure thorough removal of diseased parts and adhere to the recommended application schedule. Inadequate pruning or spacing can also contribute to disease; maintain proper practices for better air circulation."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},
'Peach___healthy': {
    'treatment': (
        "The peach trees are healthy when they show no symptoms of disease and are properly managed with appropriate irrigation, fertilization, and pest control practices."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},
'Pepper,_bell___Bacterial_spot': {
    'treatment': (
        "Remove and destroy infected plant material. Apply copper-based bactericides or antibiotics like streptomycin at the onset of symptoms and repeat as needed."
    ),
    'suggested_changes': (
        "Ensure proper spacing of plants to improve air circulation and reduce humidity. Use resistant pepper varieties if available."
    ),
    'errors_and_corrections': (
        "Failure to remove infected material or inconsistent bactericide application can lead to continued disease pressure. Ensure thorough removal and follow recommended application schedules. Inadequate spacing can also contribute to disease; maintain proper plant spacing for better air circulation."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},
'Pepper,_bell___healthy': {
    'treatment': (
        "The bell pepper plants are healthy when they show no symptoms of disease and are properly cared for with appropriate irrigation, fertilization, and pest control practices."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},
'Potato___Early_blight': {
    'treatment': (
        "Remove and destroy infected plant parts and apply fungicides like chlorothalonil or mancozeb starting at the first sign of disease, with follow-up applications every 7-10 days."
    ),
    'suggested_changes': (
        "Implement crop rotation and avoid planting potatoes in the same field year after year. Ensure proper spacing to improve air circulation and reduce humidity."
    ),
    'errors_and_corrections': (
        "Failure to remove infected plant material or inadequate fungicide application can lead to increased disease pressure. Ensure thorough removal and follow application schedules. Inconsistent rotation practices can also contribute to disease persistence; adhere to rotation recommendations."
    ),
    'environmental_conditions': (
        "pH: 5.5 - 6.0 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},
'Potato___Late_blight': {
    'treatment': (
        "Remove and destroy infected plant material and apply fungicides like metalaxyl or chlorothalonil when the disease is first observed, with follow-up applications every 7-10 days."
    ),
    'suggested_changes': (
        "Implement crop rotation and avoid planting potatoes in the same field for consecutive years. Ensure proper spacing and good drainage to reduce moisture and humidity around the plants."
    ),
    'errors_and_corrections': (
        "Failure to remove infected plant material or inconsistent fungicide application can lead to persistent disease. Ensure timely removal and thorough fungicide application. Inadequate rotation or drainage practices can also contribute to disease; adhere to recommended practices for better control."
    ),
    'environmental_conditions': (
        "pH: 5.5 - 6.0 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},
'Potato___healthy': {
    'treatment': (
        "The potato plants are healthy when they exhibit no symptoms of disease and are well-managed with proper irrigation, fertilization, and pest control practices."
    ),
    'environmental_conditions': (
        "pH: 5.5 - 6.0 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},
'Raspberry___healthy': {
    'treatment': (
        "The raspberry plants are healthy when they show no symptoms of disease and are properly maintained with adequate irrigation, fertilization, and pest control."
    ),
    'environmental_conditions': (
        "pH: 5.5 - 6.5 | Nitrogen: 60 - 80 kg/ha | Phosphorus: 20 - 40 kg/ha | Potassium: 80 - 120 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},
'Soybean___healthy': {
    'treatment': (
        "The soybean plants are healthy when they are free from disease symptoms and are well-managed with appropriate irrigation, fertilization, and pest control practices."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 40 - 80 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 80 - 120 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},
'Squash___Powdery_mildew': {
    'treatment': (
        "Remove and destroy infected plant parts. Apply fungicides like sulfur or potassium bicarbonate at the first sign of disease and repeat every 7-10 days as needed."
    ),
    'suggested_changes': (
        "Improve air circulation by proper spacing and avoid overhead watering to reduce humidity around the plants."
    ),
    'errors_and_corrections': (
        "Inadequate removal of infected parts or improper fungicide application can contribute to disease spread. Ensure thorough removal and consistent application. Excess moisture from overhead watering or poor spacing can also exacerbate the problem."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},
'Strawberry___Leaf_scorch': {
    'treatment': (
        "Remove and destroy infected leaves. Apply fungicides such as copper-based products or chlorothalonil at the first sign of disease and repeat every 7-10 days as necessary."
    ),
    'suggested_changes': (
        "Ensure proper plant spacing and avoid overhead irrigation to reduce humidity and improve air circulation around the plants."
    ),
    'errors_and_corrections': (
        "Failure to remove infected leaves or inconsistent fungicide application can lead to ongoing issues. Ensure thorough removal and follow application guidelines. Excess moisture from overhead watering can also contribute to disease spread."
    ),
    'environmental_conditions': (
        "pH: 5.5 - 6.5 | Nitrogen: 60 - 80 kg/ha | Phosphorus: 20 - 40 kg/ha | Potassium: 80 - 120 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},

'Strawberry___healthy': {
    'treatment': (
        "The strawberry plants are healthy when they show no symptoms of disease and are well-maintained with appropriate irrigation, fertilization, and pest management practices."
    ),
    'suggested_changes': (
        "Maintain appropriate irrigation, fertilization, and pest management practices to ensure plant health."
    ),
    'errors_and_corrections': (
        "N/A"
    ),
    'environmental_conditions': (
        "pH: 5.5 - 6.5 | Nitrogen: 60 - 80 kg/ha | Phosphorus: 20 - 40 kg/ha | Potassium: 80 - 120 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},

'Tomato___Bacterial_spot': {
    'treatment': (
        "Remove and destroy infected plant parts and apply copper-based bactericides or antibiotics like streptomycin at the onset of symptoms, with follow-up applications as needed."
    ),
    'suggested_changes': (
        "Improve air circulation by proper spacing and avoid overhead irrigation to reduce humidity. Use resistant tomato varieties if available."
    ),
    'errors_and_corrections': (
        "Inadequate removal of infected material or inconsistent bactericide application can lead to continued disease pressure. Ensure thorough removal and follow recommended application schedules. Poor spacing or irrigation practices can also contribute to disease; maintain proper plant management practices."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},

'Tomato___Early_blight': {
    'treatment': (
        "Remove and destroy infected plant parts and apply fungicides like chlorothalonil or mancozeb at the first sign of disease, with follow-up applications every 7-10 days."
    ),
    'suggested_changes': (
        "Implement crop rotation and avoid planting tomatoes in the same field year after year. Ensure proper spacing and good air circulation to reduce humidity."
    ),
    'errors_and_corrections': (
        "Failure to remove infected material or inadequate fungicide application can lead to increased disease pressure. Ensure timely removal and follow application schedules. Inconsistent rotation practices can also contribute to disease persistence; adhere to recommended crop rotation."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},

'Tomato___Late_blight': {
    'treatment': (
        "Remove and destroy infected plant material and apply fungicides like metalaxyl or chlorothalonil at the first sign of disease, with follow-up applications every 7-10 days."
    ),
    'suggested_changes': (
        "Implement crop rotation and avoid planting tomatoes in the same field year after year. Ensure proper spacing and good drainage to reduce moisture and humidity around the plants."
    ),
    'errors_and_corrections': (
        "Failure to remove infected plant material or inconsistent fungicide application can lead to persistent disease. Ensure timely removal and thorough fungicide application. Inadequate rotation or drainage practices can also contribute to disease; adhere to recommended practices for better control."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},

'Tomato___Leaf_Mold': {
    'treatment': (
        "Remove and destroy infected plant parts and apply fungicides like sulfur or copper-based products at the first sign of disease, with follow-up applications every 7-10 days."
    ),
    'suggested_changes': (
        "Improve air circulation by proper spacing and avoid overhead watering to reduce humidity around the plants."
    ),
    'errors_and_corrections': (
        "Inadequate removal of infected parts or improper fungicide application can lead to continued disease issues. Ensure thorough removal and consistent application. Excess moisture from overhead watering or poor plant spacing can also contribute to disease spread."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},

'Tomato___Septoria_leaf_spot': {
    'treatment': (
        "Remove and destroy infected plant material and apply fungicides like chlorothalonil or mancozeb at the first sign of disease, with follow-up applications every 7-10 days."
    ),
    'suggested_changes': (
        "Implement crop rotation and avoid planting tomatoes in the same field year after year. Ensure proper plant spacing and good air circulation to reduce humidity."
    ),
    'errors_and_corrections': (
        "Failure to remove infected material or inadequate fungicide application can lead to persistent disease. Ensure timely removal and thorough application. Inconsistent rotation practices or poor air circulation can also contribute to disease persistence; adhere to recommended practices."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},

'Tomato___Spider_mites_Two-spotted_spider_mite': {
    'treatment': (
        "Apply miticides like abamectin or spinosad to control spider mites. Ensure thorough coverage of plant foliage and repeat treatments as needed."
    ),
    'suggested_changes': (
        "Improve air circulation by proper plant spacing and avoid excessive nitrogen fertilization, which can exacerbate mite problems."
    ),
    'errors_and_corrections': (
        "Inadequate miticide application or poor coverage can lead to continued mite problems. Ensure thorough application and follow-up treatments. Excessive nitrogen can also contribute to mite infestations; manage fertilization practices accordingly."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},

'Tomato___Target_Spot': {
    'treatment': (
        "Remove and destroy infected plant parts and apply fungicides like chlorothalonil or mancozeb at the first sign of disease, with follow-up applications every 7-10 days."
    ),
    'suggested_changes': (
        "Implement crop rotation and avoid planting tomatoes in the same field year after year. Ensure proper spacing and good air circulation to reduce humidity."
    ),
    'errors_and_corrections': (
        "Failure to remove infected material or inadequate fungicide application can lead to persistent disease. Ensure timely removal and thorough application. Inconsistent rotation practices or poor air circulation can also contribute to disease persistence; follow recommended practices."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},

'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
    'treatment': (
        "Remove and destroy infected plants and control the vector (whiteflies) using insecticides such as imidacloprid or thiamethoxam."
    ),
    'suggested_changes': (
        "Use virus-resistant tomato varieties if available and manage whitefly populations through proper insecticide application and cultural practices."
    ),
    'errors_and_corrections': (
        "Inadequate removal of infected plants or poor vector control can lead to continued disease spread. Ensure thorough plant removal and effective vector management. Lack of resistant varieties or improper insecticide use can also contribute to virus persistence."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
},

'Tomato___Tomato_mosaic_virus': {
    'treatment': (
        "Remove and destroy infected plants. Control aphid populations with insecticides such as imidacloprid or thiamethoxam to prevent virus spread."
    ),
    'suggested_changes': (
        "Use virus-resistant tomato varieties if available and manage aphid populations through proper insecticide application and cultural practices."
    ),
    'errors_and_corrections': (
        "Failure to remove infected plants or inadequate aphid control can lead to ongoing disease issues. Ensure timely removal and effective aphid management. Lack of resistant varieties or improper insecticide use can also contribute to virus persistence."
    ),
    'environmental_conditions': (
        "pH: 6.0 - 6.8 | Nitrogen: 80 - 120 kg/ha | Phosphorus: 20 - 50 kg/ha | Potassium: 100 - 200 kg/ha | Humidity: 50% - 70% | Temperature: 15°C - 25°C"
    )
}



}



languages = {
    'en': 'English',
    'hi': 'Hindi',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'ta': 'Tamil'
}

selected_language = st.sidebar.selectbox("Select Language", list(languages.keys()), format_func=lambda lang: languages[lang])

st.title(translate_text("Plant Disease Prediction", selected_language))
st.sidebar.header(translate_text("Language Selection", selected_language))

upload_prompt = translate_text("Choose an image...", selected_language)
predicted_class_label = translate_text("Predicted class:", selected_language)
treatment_label = translate_text("Treatment Details", selected_language)

uploaded_file = st.file_uploader(upload_prompt, type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image.resize((256, 256))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    predicted_class_name = class_names[predicted_class]

    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write(f"{predicted_class_label} {translate_text(predicted_class_name, selected_language)}")

    disease_info = disease_treatments.get(predicted_class_name, {})

    condition = st.selectbox(
        translate_text("Select condition to view details", selected_language),
        ["Treatment", "Suggested Changes", "Possible Errors", "Environmental Conditions"]
    )

    if condition == "Treatment":
        treatment_html = f"""
        <div style="background-color: #ffffff; padding: 10px; border-radius: 5px;border: 1px solid #000000;">
            <h3 style="color: #000000;">{translate_text("Treatment", selected_language)}</h3>
            <p style="color: #000000;">{translate_text(disease_info.get('treatment', "No treatment information available."), selected_language)}</p>
        </div>
        """
        st.markdown(treatment_html, unsafe_allow_html=True)
    elif condition == "Suggested Changes":
        suggested_changes_html = f"""
        <div style="background-color: #ffffff; padding: 10px; border-radius: 5px;border: 1px solid #000000;">
            <h3 style="color: #000000;">{translate_text("Suggested Changes in Farming Technique", selected_language)}</h3>
            <p style="color: #000000;">{translate_text(disease_info.get('suggested_changes', "No information available."), selected_language)}</p>
        </div>
        """
        st.markdown(suggested_changes_html, unsafe_allow_html=True)
    elif condition == "Possible Errors":
        errors_and_corrections_html = f"""
        <div style="background-color:#ffffff; padding: 10px; border-radius: 5px;border: 1px solid #000000;">
            <h3 style="color: #000000;">{translate_text("Possible Errors and Corrections", selected_language)}</h3>
           <p style="color: #000000;">{translate_text(disease_info.get('errors_and_corrections', "No information available."), selected_language)}</p>
        </div>
        """
        st.markdown(errors_and_corrections_html, unsafe_allow_html=True)
    elif condition == "Environmental Conditions":
        environmental_conditions_html = f"""
        <div style="background-color: #ffffff; padding: 10px; border-radius: 5px;border: 1px solid #000000;">
            <h3 style="color:  #000000;">{translate_text("Preferable Environmental Conditions", selected_language)}</h3>
           <p style="color: #000000;">{translate_text(disease_info.get('environmental_conditions', "No information available."), selected_language)}</p>
        </div>
        """
        st.markdown(environmental_conditions_html, unsafe_allow_html=True)