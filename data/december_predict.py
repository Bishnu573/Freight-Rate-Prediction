import pandas as pd
import joblib
from preprocess import preprocess

# Load trained model
model = joblib.load("models/freight_model.pkl")

# Load training data (used to build lookup tables)
train = pd.read_csv("data/train_test.csv")

# Load December data
dec = pd.read_csv("data/december_chart_inputs.csv")

# -----------------------------
# Build city -> coordinates lookup
# -----------------------------
pickup_lookup = (
    train[["pickup", "pickup_lat", "pickup_lon"]]
    .drop_duplicates("pickup")
)

delivery_lookup = (
    train[["delivery", "delivery_lat", "delivery_lon"]]
    .drop_duplicates("delivery")
)

# Merge pickup coordinates
dec = dec.merge(pickup_lookup, on="pickup", how="left")

# Merge delivery coordinates
dec = dec.merge(delivery_lookup, on="delivery", how="left")

# Fill missing market features with training medians
dec["market_index"] = train["market_index"].median()
dec["quote_signal"] = train["quote_signal"].median()

# Preprocess
dec = preprocess(dec)

# Remove prediction column if it already exists
if "predicted_rate" in dec.columns:
    dec = dec.drop(columns=["predicted_rate"])

# Predict
pred = model.predict(dec)

# Save predictions
out = pd.read_csv("data/december_chart_inputs.csv")
out["predicted_rate"] = pred
out.to_csv("data/december_chart_inputs.csv", index=False)

print("Predictions saved successfully.")
print(out.head())