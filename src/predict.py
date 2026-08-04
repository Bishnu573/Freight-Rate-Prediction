import joblib
import pandas as pd

from preprocess import preprocess

# Load trained model
model = joblib.load("models/freight_model.pkl")

# Load validation data
validation = pd.read_csv("data/validation.csv")

# Save load_id
load_ids = validation["load_id"]

# Preprocess
validation = preprocess(validation)

# Remove load_id before prediction
X = validation.drop(columns=["load_id"])

# Predict
predictions = model.predict(X)

# Create submission
submission = pd.DataFrame({
    "load_id": load_ids,
    "predicted_rate": predictions
})

# Save predictions
submission.to_csv("validation_predictions.csv", index=False)

print("validation_predictions.csv created successfully!")
print(submission.head())