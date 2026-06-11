import joblib
import numpy as np
from pathlib import Path
from app.models import PatientData, PredictionResult

# --- Config ---
ML_DIR = Path("ml")

# --- Load model and scaler ---
model = joblib.load(ML_DIR / "best_model.pkl")
scaler = joblib.load(ML_DIR / "scaler.pkl")

# --- Predict ---
def predict(patient: PatientData) -> PredictionResult:
    features = np.array([[
        patient.pregnancies,
        patient.glucose,
        patient.blood_pressure,
        patient.skin_thickness,
        patient.insulin,
        patient.bmi,
        patient.diabetes_pedigree,
        patient.age
    ]])

    features_scaled = scaler.transform(features)
    prediction = int(model.predict(features_scaled)[0])
    probability = float(model.predict_proba(features_scaled)[0][1])

    diagnosis = "Diabetes detected" if prediction == 1 else "No diabetes detected"

    return PredictionResult(
        prediction=prediction,
        probability=round(probability, 3),
        diagnosis=diagnosis
    )