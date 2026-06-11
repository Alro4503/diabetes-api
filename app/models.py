from pydantic import BaseModel, Field

# --- Input model: patient data for prediction ---
class PatientData(BaseModel):
    pregnancies: float = Field(..., ge=0)
    glucose: float = Field(..., gt=0)
    blood_pressure: float = Field(..., gt=0)
    skin_thickness: float = Field(..., ge=0)
    insulin: float = Field(..., ge=0)
    bmi: float = Field(..., gt=0)
    diabetes_pedigree: float = Field(..., gt=0)
    age: float = Field(..., gt=0)

# --- Output model: prediction result ---
class PredictionResult(BaseModel):
    prediction: int
    probability: float
    diagnosis: str

# --- Input model: chat message ---
class ChatMessage(BaseModel):
    message: str
    prediction_context: PredictionResult | None = None

# --- Output model: chat response ---
class ChatResponse(BaseModel):
    response: str