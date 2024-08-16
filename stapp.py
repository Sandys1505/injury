import streamlit as st
import pickle
import numpy as np
import joblib
import pandas as pd

model = joblib.load('model.pkl')

def prediction(input_data):
       feature_names=['cross_street_type',' weather', 'surface_condition', 'traffic_control', 'vehicle_movement', 'speed_limit']
       input_data_df = pd.DataFrame([input_data], columns=feature_names)
        # Get prediction
       prediction = model.predict(input_data)
# App header
st.title("Crash Predictor")

def main():
       cross_street_type = st.number_input("Cross-Street Type", options=['County', 'Municipality', 'Maryland (State)', 'US (State)',
       'Unknown', 'Other Public Roadway', 'Ramp', 'Government',
       'Interstate (State)', 'Service Road'])
       weather = st.number_input("Weather", options=['CLOUDY', 'CLEAR', 'RAINING', 'UNKNOWN', 'FOGGY', 'OTHER',
       'SNOW', 'SLEET', 'WINTRY MIX', 'BLOWING SNOW', 'SEVERE WINDS',
       'BLOWING SAND, SOIL, DIRT'])
       surface_condition = st.number_input("Surface Condition", options=['DRY', 'WET', 'UNKNOWN', 'WATER(STANDING/MOVING)',
       'MUD, DIRT, GRAVEL', 'ICE', 'SNOW', 'SLUSH', 'OTHER', 'OIL',
       'SAND'])
       traffic_control = st.number_input("Traffic Control", options=['NO CONTROLS', 'TRAFFIC SIGNAL', 'YIELD SIGN','OTHER',
       'STOP SIGN', 'FLASHING TRAFFIC SIGNAL', 'WARNING SIGN', 'UNKNOWN',
       'PERSON', 'SCHOOL ZONE SIGN DEVICE', 'RAILWAY CROSSING DEVICE'])
       vehicle_movement = st.number_input("Vehicle Movement", options=['MOVING CONSTANT SPEED', 'MAKING LEFT TURN', 'ACCELERATING',
       'UNKNOWN', 'PARKED', 'SLOWING OR STOPPING',
       'STOPPED IN TRAFFIC LANE', 'PARKING', 'MAKING RIGHT TURN',
       'STARTING FROM LANE', 'BACKING', 'CHANGING LANES', 'MAKING U TURN',
    'PASSING', 'STARTING FROM PARKED', 'DRIVERLESS MOVING VEH.',
       'LEAVING TRAFFIC LANE', 'ENTERING TRAFFIC LANE', 'OTHER',
       'NEGOTIATING A CURVE', 'RIGHT TURN ON RED', 'SKIDDING'])
       speed_limit = st.number_input("Speed Limit", min_value=0,max_value=75)

# Prediction button
if st.button("Predict"):
    try:
        
        # Map the prediction to its corresponding injury category
        injury_map = {
            3: "FATAL INJURY",
            1: "NO APPARENT INJURY",
            2: "SUSPECTED MINOR INJURY",
            4: "POSSIBLE INJURY",
            0: "SUSPECTED SERIOUS INJURY"
        }
        
        injury_category = injury_map.get(prediction, "Unknown")
        
        # Display the prediction result
        st.write(f"Prediction: {injury_category}")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
     main()