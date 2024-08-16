import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model, scaler, and feature names
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

# Define injury categories for interpretation
injury_labels = {
    0: "SUSPECTED SERIOUS INJURY",
    1: "NO APPARENT INJURY",
    2: "SUSPECTED MINOR INJURY",
    3: "POSSIBLE INJURY",
    4: "FATAL INJURY"
}

# Define feature value mappings
cross_street_types = ['County', 'Municipality', 'Maryland (State)', 'US (State)', 'Unknown',
                      'Other Public Roadway', 'Ramp', 'Government', 'Interstate (State)', 'Service Road']
collision_types = ['OPPOSITE DIRECTION SIDESWIPE', 'STRAIGHT MOVEMENT ANGLE', 'SAME DIR REAR END',
                   'SINGLE VEHICLE', 'OTHER', 'SAME DIRECTION SIDESWIPE', 'HEAD ON LEFT TURN', 'HEAD ON',
                   'ANGLE MEETS LEFT HEAD ON', 'SAME DIRECTION RIGHT TURN', 'ANGLE MEETS LEFT TURN',
                   'SAME DIR BOTH LEFT TURN', 'ANGLE MEETS RIGHT TURN', 'SAME DIR REND RIGHT TURN',
                   'SAME DIRECTION LEFT TURN', 'SAME DIR REND LEFT TURN', 'UNKNOWN', 'OPPOSITE DIR BOTH LEFT TURN']
weather_conditions = ['CLOUDY', 'CLEAR', 'RAINING', 'UNKNOWN', 'FOGGY', 'OTHER', 'SNOW', 'SLEET',
                      'WINTRY MIX', 'BLOWING SNOW', 'SEVERE WINDS', 'BLOWING SAND, SOIL, DIRT']
surface_conditions = ['DRY', 'WET', 'UNKNOWN', 'WATER(STANDING/MOVING)', 'MUD, DIRT, GRAVEL', 'ICE',
                      'SNOW', 'SLUSH', 'OTHER', 'OIL', 'SAND']
lights = ['DAYLIGHT', 'DARK LIGHTS ON', 'DARK NO LIGHTS', 'DUSK', 'DARK -- UNKNOWN LIGHTING', 'DAWN',
          'OTHER', 'UNKNOWN']
traffic_controls = ['NO CONTROLS', 'TRAFFIC SIGNAL', 'YIELD SIGN', 'OTHER', 'STOP SIGN',
                    'FLASHING TRAFFIC SIGNAL', 'WARNING SIGN', 'UNKNOWN', 'PERSON',
                    'SCHOOL ZONE SIGN DEVICE', 'RAILWAY CROSSING DEVICE']
driver_substance_abuses = ['NONE DETECTED', 'UNKNOWN', 'ALCOHOL PRESENT', 'COMBINED SUBSTANCE PRESENT',
                           'ILLEGAL DRUG PRESENT', 'ALCOHOL CONTRIBUTED', 'ILLEGAL DRUG CONTRIBUTED',
                           'MEDICATION CONTRIBUTED', 'MEDICATION PRESENT', 'COMBINATION CONTRIBUTED', 'OTHER']
driver_at_faults = ['Yes', 'No', 'Unknown']
driver_distracted_bys = ['LOOKED BUT DID NOT SEE', 'NOT DISTRACTED', 'UNKNOWN', 'INATTENTIVE OR LOST IN THOUGHT',
                         'OTHER DISTRACTION', 'USING OTHER DEVICE CONTROLS INTEGRAL TO VEHICLE', 'NO DRIVER PRESENT',
                         'TEXTING FROM A CELLULAR PHONE', 'EATING OR DRINKING', 'DISTRACTED BY OUTSIDE PERSON OBJECT OR EVENT',
                         'OTHER CELLULAR PHONE RELATED', 'BY MOVING OBJECT IN VEHICLE', 'TALKING OR LISTENING TO CELLULAR PHONE',
                         'BY OTHER OCCUPANTS', 'ADJUSTING AUDIO AND OR CLIMATE CONTROLS',
                         'OTHER ELECTRONIC DEVICE (NAVIGATIONAL PALM PILOT)', 'DIALING CELLULAR PHONE',
                         'USING DEVICE OBJECT BROUGHT INTO VEHICLE', 'SMOKING RELATED']
vehicle_damage_extents = ['FUNCTIONAL', 'DISABLING', 'NO DAMAGE', 'UNKNOWN', 'SUPERFICIAL', 'DESTROYED', 'OTHER']
vehicle_movements = ['MOVING CONSTANT SPEED', 'MAKING LEFT TURN', 'ACCELERATING', 'UNKNOWN', 'PARKED',
                     'SLOWING OR STOPPING', 'STOPPED IN TRAFFIC LANE', 'PARKING', 'MAKING RIGHT TURN',
                     'STARTING FROM LANE', 'BACKING', 'CHANGING LANES', 'MAKING U TURN', 'PASSING',
                     'STARTING FROM PARKED', 'DRIVERLESS MOVING VEH.', 'LEAVING TRAFFIC LANE',
                     'ENTERING TRAFFIC LANE', 'OTHER', 'NEGOTIATING A CURVE', 'RIGHT TURN ON RED', 'SKIDDING']
vehicle_continuing_dirs = ['South', 'North', 'East', 'West', 'Unknown']
vehicle_going_dirs = ['South', 'West', 'North', 'East', 'Unknown']
speed_limits = [0, 35, 40, 20, 10, 25, 45, 15, 30, 50, 5, 55, 65, 60, 75, 70]

def preprocess_input(data):
    df = pd.DataFrame([data])
    
    # Fill missing values with default values
    df.fillna({
        'Cross-Street Type': 'Unknown',
        'Collision Type': 'OTHER',
        'Weather': 'UNKNOWN',
        'Surface Condition': 'UNKNOWN',
        'Light': 'UNKNOWN',
        'Traffic Control': 'NO CONTROLS',
        'Driver Substance Abuse': 'NONE DETECTED',
        'Driver At Fault': 'Unknown',
        'Driver Distracted By': 'NOT DISTRACTED',
        'Vehicle Damage Extent': 'NO DAMAGE',
        'Vehicle Movement': 'UNKNOWN',
        'Vehicle Continuing Dir': 'Unknown',
        'Vehicle Going Dir': 'Unknown',
        'Speed Limit': 0
    }, inplace=True)
    
    # Encoding categorical features
    df['Cross-Street Type'] = pd.Categorical(df['Cross-Street Type'], categories=cross_street_types).codes
    df['Collision Type'] = pd.Categorical(df['Collision Type'], categories=collision_types).codes
    df['Weather'] = pd.Categorical(df['Weather'], categories=weather_conditions).codes
    df['Surface Condition'] = pd.Categorical(df['Surface Condition'], categories=surface_conditions).codes
    df['Light'] = pd.Categorical(df['Light'], categories=lights).codes
    df['Traffic Control'] = pd.Categorical(df['Traffic Control'], categories=traffic_controls).codes
    df['Driver Substance Abuse'] = pd.Categorical(df['Driver Substance Abuse'], categories=driver_substance_abuses).codes
    df['Driver At Fault'] = pd.Categorical(df['Driver At Fault'], categories=driver_at_faults).codes
    df['Driver Distracted By'] = pd.Categorical(df['Driver Distracted By'], categories=driver_distracted_bys).codes
    df['Vehicle Damage Extent'] = pd.Categorical(df['Vehicle Damage Extent'], categories=vehicle_damage_extents).codes
    df['Vehicle Movement'] = pd.Categorical(df['Vehicle Movement'], categories=vehicle_movements).codes
    df['Vehicle Continuing Dir'] = pd.Categorical(df['Vehicle Continuing Dir'], categories=vehicle_continuing_dirs).codes
    df['Vehicle Going Dir'] = pd.Categorical(df['Vehicle Going Dir'], categories=vehicle_going_dirs).codes
    
    # Ensure all expected columns are present
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Reorder columns to match the model's input features
    df = df[features]

    # Scale features
    df_scaled = scaler.transform(df)
    
    return df_scaled

def prediction(input_data):
    # Process input data
    input_data_df = preprocess_input(input_data)
    
    # Predict using the model
    prediction = model.predict(input_data_df)
    
    # Map the prediction to readable format
    injury_category = injury_labels.get(prediction[0], 'UNKNOWN')
    return injury_category

def main():
    st.title('Injury Severity Prediction')
    
    # Define input fields
    cross_street_type = st.selectbox('Cross-Street Type', cross_street_types)
    collision_type = st.selectbox('Collision Type', collision_types)
    weather = st.selectbox('Weather', weather_conditions)
    surface_condition = st.selectbox('Surface Condition', surface_conditions)
    light = st.selectbox('Light', lights)
    traffic_control = st.selectbox('Traffic Control', traffic_controls)
    driver_substance_abuse = st.selectbox('Driver Substance Abuse', driver_substance_abuses)
    driver_at_fault = st.selectbox('Driver At Fault', driver_at_faults)
    driver_distracted_by = st.selectbox('Driver Distracted By', driver_distracted_bys)
    vehicle_damage_extent = st.selectbox('Vehicle Damage Extent', vehicle_damage_extents)
    vehicle_movement = st.selectbox('Vehicle Movement', vehicle_movements)
    vehicle_continuing_dir = st.selectbox('Vehicle Continuing Dir', vehicle_continuing_dirs)
    vehicle_going_dir = st.selectbox('Vehicle Going Dir', vehicle_going_dirs)
    speed_limit = st.slider('Speed Limit', min_value=min(speed_limits), max_value=max(speed_limits))
    
    # Collect input data
    input_data = {
        'Cross-Street Type': cross_street_type,
        'Collision Type': collision_type,
        'Weather': weather,
        'Surface Condition': surface_condition,
        'Light': light,
        'Traffic Control': traffic_control,
        'Driver Substance Abuse': driver_substance_abuse,
        'Driver At Fault': driver_at_fault,
        'Driver Distracted By': driver_distracted_by,
        'Vehicle Damage Extent': vehicle_damage_extent,
        'Vehicle Movement': vehicle_movement,
        'Vehicle Continuing Dir': vehicle_continuing_dir,
        'Vehicle Going Dir': vehicle_going_dir,
        'Speed Limit': speed_limit
    }
    
    # Make prediction
    if st.button('Predict Injury Severity'):
        result = prediction(input_data)
        st.write(f'Predicted Injury Severity: {result}')

if __name__ == "__main__":
    main()
