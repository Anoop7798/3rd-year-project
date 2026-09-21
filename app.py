import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "model/car_price_model.pkl"

st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Used Car Price Prediction")
st.write("Enter the details of your used car to estimate its selling price.")

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

name = st.text_input("Car Name", "Maruti Swift Dzire VDI")
year = st.number_input("Manufacturing Year", 1990, 2026, 2018)
km_driven = st.number_input("Kilometers Driven", 0, 1000000, 50000)

fuel = st.selectbox(
    "Fuel Type",
    ["Diesel", "Petrol", "CNG", "LPG", "Electric"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Individual", "Dealer", "Trustmark Dealer"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

owner = st.selectbox(
    "Owner Type",
    [
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
        "Test Drive Car"
    ]
)

mileage = st.number_input("Mileage (km/l)", 0.0, 100.0, 18.0)
engine = st.number_input("Engine (CC)", 500.0, 10000.0, 1200.0)
max_power = st.number_input("Max Power (bhp)", 0.0, 2000.0, 80.0)
seats = st.number_input("Number of Seats", 2, 15, 5)

if st.button("💰 Predict Car Price"):
    input_data = pd.DataFrame({
        "name": [name],
        "year": [year],
        "km_driven": [km_driven],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner],
        "mileage": [mileage],
        "engine": [engine],
        "max_power": [max_power],
        "seats": [seats]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Selling Price: ₹{prediction:,.0f}")
    st.info("This is an ML-based estimate. Actual market price may vary.")
