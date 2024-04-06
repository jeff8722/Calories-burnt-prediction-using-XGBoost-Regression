import streamlit as st
import xgboost as xgb
import pandas as pd
calories = 0
# Load the XGBoost model
model = xgb.Booster()
model.load_model('Model/my_model.json')

# Define the input fields
gender_mapping = {'Male': 0, 'Female': 1}
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=18, max_value=100, value=30)
height = st.number_input('Height (cm)', min_value=100, max_value=250, value=170)
weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=70)
duration = st.number_input('Duration (minutes)', min_value=10, max_value=120, value=30)
heart_rate = st.number_input('Heart Rate (bpm)', min_value=60, max_value=200, value=80)
body_temp = st.number_input('Body Temp (°C)', min_value=35, max_value=42, value=37)

# Add a Calculate button
if st.button('Calculate'):
    # Convert the input values to a DataFrame
    input_data = pd.DataFrame({
        'Gender': [gender_mapping[gender]],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'Duration': [duration],
        'Heart_Rate': [heart_rate],
        'Body_Temp': [body_temp],
        #'Calories': [0] # Set Calories to 0 as it will be predicted by the model
    })

    # Pass the input data through the XGBoost model
    prediction = model.predict(xgb.DMatrix(input_data))

    # Display the predicted calories burnt
    st.write('Predicted Calories Burnt:', int(prediction[0]))

# Display the input value of calories
#st.write('Calories:', calories)