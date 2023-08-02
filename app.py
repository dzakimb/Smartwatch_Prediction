import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import joblib

URL = "clean_data.csv"
MODEL_URL = "finalized_model.sav"

model = joblib.load(MODEL_URL)

def model_predict(model, input_dict):
    OFFSET = 41.50
    input_df = pd.DataFrame([input_dict])
    result = model.predict(input_df)
    upper_bound = result[0] + OFFSET/2
    lower_bound = result[0] - OFFSET/2
    return f"Price is in range of {lower_bound:.2f} - {upper_bound:.2f}$"






st.title("Smart Watch Price Prediction")
st.write("This is an application to predict price of smartwatch given its features. Build using Python")

data = pd.read_csv(URL)
data.drop("Unnamed: 0", axis=1, inplace=True)
st.subheader("How's data look like")
st.write(data.head(5))

CAT_FEAT = ['Brand', 'Operating System', 'GPS']
NUM_FEAT = ['Display Size (inches)','Water Resistance (meters)','Battery Life (days)','new_res']
FEAT = CAT_FEAT + NUM_FEAT
TARGET = ['Price (USD)']


st.subheader("Make a dream smartwatch")
st.write("Let's choose the features of your smartwatch and get the price!")



form = st.form("Form")
with form:
    brand_chosen = st.selectbox("Choose your brand", data['Brand'].value_counts().index.to_list())
    os_chosen = st.selectbox("Choose your OS do you want", data['Operating System'].value_counts().index.to_list())
    gps_chosen = st.selectbox('do you want this smartwatch has a GPS?', ("Yes", "No"))
    cat_input = [brand_chosen, os_chosen, gps_chosen]

    display_size = st.slider("How many inches display do you want?", min_value=0.9, max_value=4.0, value=2.0)
    battery_life = st.slider("How long battery should last (Days)?", min_value=1.0, max_value=72.0, value=14.0)
    water_res = st.slider("How many meters watch should last under water?", min_value=1.5, max_value=200.0, value=100.0)
    width_res = st.slider("How many width resolution you need?", min_value=126.0, max_value=960.0 , value=454.0)
    height_res = st.slider("How many height resolution you need?", min_value=36.0, max_value=484.0 , value=454.0)
    resolution = width_res * height_res
    num_input = [display_size, water_res, battery_life, resolution]
    
    feat_input = cat_input + num_input
    input_dict = {k:v for (k,v) in zip(FEAT, feat_input)}
    input_df = pd.DataFrame([input_dict])
    
    

    submit_form = st.form_submit_button("Submit")

    if submit_form:

        st.write(input_df)
        price_result = model_predict(model, input_dict)
        st.write(price_result)





