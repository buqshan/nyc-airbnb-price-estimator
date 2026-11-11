# 🗽 NYC Airbnb Fair Price Estimator

A small Streamlit web app that estimates a **fair nightly price** for an Airbnb listing in New York City and shows a **typical price range** around that estimate.

**Live app:** https://nyc-airbnb-price-buqshan.streamlit.app/

> Built by **Abdullah A. Buqshan** as a personal learning artefact for the *Machine Learning* module of the MSc Artificial Intelligence at the University of Essex Online (2026).

---

## What the app does

You choose:

| Input | What it means |
|---|---|
| **Borough** and **neighbourhood** | Where the listing is. The app converts this into the distance to Midtown Manhattan. |
| **Room type** | Entire home/apartment, private room or shared room |
| **Reviews per month** | How active and popular the listing is |
| **Availability (days per year)** | How many days a year the listing can be booked |

The app returns:

- ✅ **Estimated fair price** per night (the model's median prediction)
- ✅ **Typical range**: the middle 50% of prices for similar listings (25th to 75th percentile)

---

## How it works

- **Data:** New York City Airbnb Open Data (2019), about 48,900 listings, cleaned in a team project.
- **Target:** the natural log of the nightly price, ln(price), because prices are strongly right-skewed. Predictions are converted back to dollars with exp().
- **Features:**
  - Room type (one-hot encoded)
  - Distance to Midtown (km)
  - Reviews per month (ln(1 + x) transformed)
  - Availability (days per year)
- **Model:** scikit-learn `HistGradientBoostingRegressor` inside a `Pipeline`:
  - One model predicts the **median** price.
  - Two **quantile** models (q = 0.25 and q = 0.75) produce the typical range.
- **Evaluation:** a **host-grouped** train/test split, so no host appears in both sets. This tests the model on **unseen hosts**, which is closer to real use.

---

## Version 2 performance (test set, unseen hosts)

| Metric | Value |
|---|---|
| R² (on ln(price)) | 0.54 |
| Mean absolute error (MAE) | $63.1 |
| **Median absolute error** | **$26.8** |
| Typical-range coverage (target 50%) | 48% |

**How to read this:** for half of the listings, the estimate is within about **$27** of the real price. The MAE is higher because a small number of luxury listings are very hard to predict.

---

## Version history

| Version | Changes |
|---|---|
| **v1** | First release (HistGradientBoosting): earlier cleaned dataset (48,396 listings), random 80/20 split, ln(1 + price) target; test R² 0.58, MAE $46.4; range shown as ± $46 |
| **v2** | Updated cleaned dataset (about 48,900 listings); host-grouped evaluation (unseen hosts); ln(price) target; new **typical price range** from quantile regression; borough and neighbourhood lookup |

---

## Repository files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `price_model.joblib` | Trained model bundle (median, low and high models, plus median error) |
| `neighbourhood_distance.csv` | Lookup table: borough and neighbourhood → distance to Midtown (km) |
| `requirements.txt` | Python packages needed to run the app |

---

## Run it on your own computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## Limitations

- ⚠️ The data is from **2019**. Prices, demand and NYC short-term rental rules have changed since then (for example, Local Law 18, enforced from September 2023).
- ⚠️ The model uses only four features. It does not know about photos, amenities, reviews' content, size or season.
- ⚠️ Estimates are **guidance only**, not a valuation or pricing advice.

---

## Acknowledgements

- Dataset: *New York City Airbnb Open Data* (Inside Airbnb, via Kaggle).
- Built with Python, pandas, scikit-learn and Streamlit.
- Géron, A. (2025) *Hands-On Machine Learning with Scikit-Learn and PyTorch*. Sebastopol, CA: O'Reilly Media.
