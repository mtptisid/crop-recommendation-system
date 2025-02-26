import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("crop_data.csv")

# Convert column names (remove spaces)
data.columns = data.columns.str.strip()

# Encode categorical features (Soil Type, Crop Type, Fertilizer Name)
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fertilizer = LabelEncoder()

data["Soil Type"] = le_soil.fit_transform(data["Soil Type"].astype(str))
data["Crop Type"] = le_crop.fit_transform(data["Crop Type"].astype(str))
data["Fertilizer Name"] = le_fertilizer.fit_transform(data["Fertilizer Name"].astype(str))

# Features & Target
X = data[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
y = data['Crop Type']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model
with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load Model for Prediction
with open("crop_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Prediction Function
def predict_crop(temp, humidity, moisture, soil_type, nitrogen, potassium, phosphorous):
    try:
        soil_encoded = le_soil.transform([soil_type])[0]  # Encode soil type
    except ValueError:
        return "Unknown Soil Type - No Recommendation Available"

    input_features = np.array([[temp, humidity, moisture, soil_encoded, nitrogen, potassium, phosphorous]])
    predicted_crop_code = loaded_model.predict(input_features)[0]
    predicted_crop = le_crop.inverse_transform([predicted_crop_code])[0]
    return predicted_crop

# Streamlit UI
st.title("🌱 Crop Recommendation System")

# User Inputs
temp = st.number_input("Temperature (°C)", min_value=10.0, max_value=50.0, value=30.0)
humidity = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0, value=60.0)
moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=40.0)
soil_type = st.selectbox("Soil Type", list(le_soil.classes_))
nitrogen = st.number_input("Nitrogen Level", min_value=0, max_value=100, value=20)
potassium = st.number_input("Potassium Level", min_value=0, max_value=100, value=10)
phosphorous = st.number_input("Phosphorous Level", min_value=0, max_value=100, value=15)

# Predict Button
if st.button("Recommend Crop"):
    result = predict_crop(temp, humidity, moisture, soil_type, nitrogen, potassium, phosphorous)
    st.success(f"🌾 Recommended Crop: **{result}**")