"""
FastAPI Application for Wine Quality Prediction
Student: Sricharan | Roll No: 2022BCS0111
Endpoints: /health (GET), /predict (POST)
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# ── Load trained model ─────────────────────────────────────────────────────
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
model = joblib.load(model_path)

app = FastAPI(
    title="Wine Quality Prediction API",
    description="MLOps End-to-End Assignment - Sricharan (2022BCS0111)",
    version="1.0.0",
)


# ── Input schema ───────────────────────────────────────────────────────────
class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


# ── Health Check Endpoint ──────────────────────────────────────────────────
@app.get("/health")
def health():
    """Health check endpoint — returns student details."""
    return {
        "status": "healthy",
        "name": "Sricharan",
        "roll_no": "2022BCS0111",
    }


# ── Prediction Endpoint ───────────────────────────────────────────────────
@app.post("/predict")
def predict(data: WineInput):
    """Predict wine quality from input features."""
    features = np.array([[
        data.fixed_acidity,
        data.volatile_acidity,
        data.citric_acid,
        data.residual_sugar,
        data.chlorides,
        data.free_sulfur_dioxide,
        data.total_sulfur_dioxide,
        data.density,
        data.pH,
        data.sulphates,
        data.alcohol,
    ]])

    prediction = model.predict(features)[0]

    return {
        "prediction": round(float(prediction), 2),
        "name": "Sricharan",
        "roll_no": "2022BCS0111",
    }
