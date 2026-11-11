import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="NYC Airbnb Fair Price Estimator", page_icon="🏠")
bundle = joblib.load("price_model.joblib")
lookup = pd.read_csv("neighbourhood_distance.csv")

st.title("🏠 NYC Airbnb Fair Price Estimator")
st.caption("Estimates a typical nightly price from 2019 New York City listings data "
           f"(evaluated on hosts unseen during training; median error about ${bundle['median_error']:.0f}). "
           "For guidance only.")

room = st.selectbox("Room type", ["Entire home/apt", "Private room", "Shared room"])
borough = st.selectbox("Borough", sorted(lookup['neighbourhood_group'].unique()))
area = st.selectbox("Neighbourhood",
                    sorted(lookup.loc[lookup['neighbourhood_group'] == borough, 'neighbourhood']))
distance = float(lookup.loc[(lookup['neighbourhood_group'] == borough) &
                            (lookup['neighbourhood'] == area), 'distance_to_midtown_km'].iloc[0])
st.write(f"📍 About **{distance} km** from Midtown Manhattan")

reviews = st.slider("Reviews per month (demand)", 0.0, 10.0, 1.0, 0.1)
availability = st.slider("Availability (days per year)", 0, 365, 180)

if st.button("Estimate price"):
    X = pd.DataFrame([{'room_type': room, 'distance_to_midtown_km': distance,
                       'reviews_per_month_filled': reviews, 'availability_365': availability}])
    price = np.exp(bundle['median'].predict(X)[0])
    low = np.exp(bundle['low'].predict(X)[0])
    high = np.exp(bundle['high'].predict(X)[0])
    low, high = min(low, price), max(high, price)   # keep the estimate inside the range
    st.metric("Estimated fair price per night", f"${price:,.0f}")
    st.caption(f"Typical range: **\\${low:,.0f} – \\${high:,.0f}** "
               "(the middle 50% of similar listings in 2019).")
