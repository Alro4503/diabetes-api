from fastapi import FastAPI, HTTPException
from app.models import PatientData, PredictionResult, ChatMessage, ChatResponse
from app.predict import predict
from app.chat import chat

# --- App init ---
app = FastAPI(
    title="Diabetes API",
    description="REST API for diabetes prediction and AI-powered medical assistant",
    version="1.0.0"
)

# --- Health check ---
@app.get("/health")
def health():
    return {"status": "ok"}

# --- Prediction endpoint ---
@app.post("/predict", response_model=PredictionResult)
def predict_endpoint(patient: PatientData):
    try:
        return predict(patient)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Chat endpoint ---
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(message: ChatMessage):
    try:
        return chat(message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))