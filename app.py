import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="NYC Airbnb Fair Price Estimator", page_icon="🏠")
model = joblib.load("price_model.joblib")
lookup = pd.read_csv("neighbourhood_distance.csv")
MAE = 46.4  # ← replace with your test MAE from Step 1

st.title("🏠 NYC Airbnb Fair Price Estimator")
st.caption("Estimates a typical nightly price from 2019 New York City listings data. For guidance only.")

room = st.selectbox("Room type", ["Entire home/apt", "Private room", "Shared room"])
borough = st.selectbox("Borough", sorted(lookup['neighbourhood_group'].unique()))
area = st.selectbox("Neighbourhood",
                    sorted(lookup.loc[lookup['neighbourhood_group'] == borough, 'neighbourhood']))
distance = float(lookup.loc[lookup['neighbourhood'] == area, 'distance_to_midtown'].iloc[0])
st.write(f"📍 About **{distance} km** from Midtown Manhattan")

reviews = st.slider("Reviews per month (demand)", 0.0, 10.0, 1.0, 0.1)
availability = st.slider("Availability (days per year)", 0, 365, 180)

if st.button("Estimate price"):
    X = pd.DataFrame([{'room_type': room, 'distance_to_midtown': distance,
                       'reviews_per_month': reviews, 'availability_365': availability}])
    price = np.expm1(model.predict(X)[0])
    st.metric("Estimated fair price per night", f"${price:,.0f}")
    st.caption(f"Typical range: ${max(price - MAE, 10):,.0f} – ${price + MAE:,.0f} "
               f"(based on the model's average error of about ${MAE}).")
