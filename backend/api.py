from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# Initialize FastAPI app
app = FastAPI(title="Credit Card Fraud Detection API")


# Input schema
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

    model_config = {"extra": "forbid"}  # reject unknown fields (strict)


# Load model and preprocessing objects
_model_path = Path(__file__).resolve().parent.parent / "models" / "fraud_detector_xgboost_v1.joblib"
model = joblib.load(_model_path)


# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Credit Card Fraud Detection API", "version": "1.0"}


# Prediction endpoint
@app.post("/predict")
def predict(transaction: Transaction) -> dict:
    """Accept one transaction, return fraud probability and class."""

    # convert transaction to dataframe and ensure column order
    df = pd.DataFrame([transaction.model_dump()])
    
    # predict probability of fraud
    proba = model.predict_proba(df)[0, 1]
    pred = 1 if proba >= 0.5 else 0

    # return prediction
    return {
        "fraud_probability": float(round(proba, 4)),
        "predicted_class": int(pred)
    }

## example to test prediction
'''
{
  "V1": -5.100255663,
  "V2": 3.63344238,
  "V3": -3.843918624,
  "V4": 0.183208447,
  "V5": -1.183997376,
  "V6": 1.602139333,
  "V7": -3.005953152,
  "V8": -8.645037802,
  "V9": 1.285458107,
  "V10": -3.717481375,
  "V11": 3.719211539,
  "V12": -5.034029747,
  "V13": 0.918999158,
  "V14": -4.220365918,
  "V15": -1.050499957,
  "V16": -1.691045268,
  "V17": -2.372423207,
  "V18": 0.450098503,
  "V19": 0.407805061,
  "V20": -2.806301923,
  "V21": 8.280439326,
  "V22": -2.797149541,
  "V23": 1.090706593,
  "V24": -0.159259597,
  "V25": 0.532156369,
  "V26": -0.497125808,
  "V27": 0.943621625,
  "V28": 0.553580692,
  "Amount": 261.22
}
'''


# Batch prediction endpoint
@app.post("/predict_batch")
def predict_batch(transactions: list[Transaction]) -> list[dict]:
    """Accept a list of transactions, return list of predictions."""

    # convert transaction to dataframe and ensure column order
    df = pd.DataFrame([transaction.model_dump() for transaction in transactions])

    # predict probability of fraud
    proba = model.predict_proba(df)[:, 1]
    pred = model.predict(df)

    # return predictions
    return [
        {
            "fraud_probability": float(round(p, 4)),
            "predicted_class": int(c),
        }
        for p, c in zip(proba, pred)
    ]

## example to test batch prediction
'''
[
{
  "V1": -5.100255663,
  "V2": 3.63344238,
  "V3": -3.843918624,
  "V4": 0.183208447,
  "V5": -1.183997376,
  "V6": 1.602139333,
  "V7": -3.005953152,
  "V8": -8.645037802,
  "V9": 1.285458107,
  "V10": -3.717481375,
  "V11": 3.719211539,
  "V12": -5.034029747,
  "V13": 0.918999158,
  "V14": -4.220365918,
  "V15": -1.050499957,
  "V16": -1.691045268,
  "V17": -2.372423207,
  "V18": 0.450098503,
  "V19": 0.407805061,
  "V20": -2.806301923,
  "V21": 8.280439326,
  "V22": -2.797149541,
  "V23": 1.090706593,
  "V24": -0.159259597,
  "V25": 0.532156369,
  "V26": -0.497125808,
  "V27": 0.943621625,
  "V28": 0.553580692,
  "Amount": 261.22
},
{
  "V1": -0.85978107,
  "V2": 1.032287911,
  "V3": 1.962003913,
  "V4": 1.078845991,
  "V5": 0.542301878,
  "V6": -0.38579498,
  "V7": 0.470521246,
  "V8": -0.142605503,
  "V9": 0.592896407,
  "V10": -0.583115882,
  "V11": 0.105000776,
  "V12": -2.844714447,
  "V13": 1.635223825,
  "V14": 1.674877735,
  "V15": 0.568444788,
  "V16": 0.035558179,
  "V17": 0.360920367,
  "V18": 0.26633733,
  "V19": 0.599229964,
  "V20": -0.052183391,
  "V21": -0.44859547,
  "V22": -1.124339247,
  "V23": -0.202307559,
  "V24": -0.230251202,
  "V25": 0.117740608,
  "V26": -0.595905267,
  "V27": -0.028956594,
  "V28": 0.053313974,
  "Amount": 21.05
}
]
'''


# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)