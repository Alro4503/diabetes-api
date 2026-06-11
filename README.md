# Diabetes API

REST API built with FastAPI to serve a diabetes prediction model and an AI-powered
medical assistant. Integrates a trained scikit-learn Random Forest with the Groq API
(Llama 3.1 8B). Dockerized for reproducibility.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?logo=fastapi&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama_3.1_8B-orange)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker&logoColor=white)

---

## Table of Contents

- [Abstract](#abstract)
- [Architecture](#architecture)
- [Endpoints](#endpoints)
- [Development](#development)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Abstract

This API serves as the inference layer for the [diabetes-classifier](https://github.com/Alro4503/diabetes-classifier) project.
It exposes two endpoints: a prediction endpoint that loads a trained Random Forest
model to classify diabetes risk from patient data, and a chat endpoint powered by
Llama 3.1 8B via Groq that provides natural language explanations of the results.

---

## Architecture

    Patient data (JSON)
         ↓
    /predict endpoint
         ↓
    StandardScaler → Random Forest (sklearn)
         ↓
    PredictionResult (prediction, probability, diagnosis)
         ↓
    /chat endpoint (optional)
         ↓
    Llama 3.1 8B via Groq API
         ↓
    Natural language explanation

---

## Endpoints

### GET /health
Health check. Returns `{"status": "ok"}` if the API is running.

### POST /predict
Receives patient data and returns a diabetes prediction.

**Request body:**

    {
      "pregnancies": 2,
      "glucose": 148,
      "blood_pressure": 72,
      "skin_thickness": 35,
      "insulin": 100,
      "bmi": 33.6,
      "diabetes_pedigree": 0.627,
      "age": 50
    }

**Response:**

    {
      "prediction": 1,
      "probability": 0.72,
      "diagnosis": "Diabetes detected"
    }

### POST /chat
Receives a message and optionally the prediction context. Returns a natural language
explanation from Llama 3.1 8B. Responds in the same language the user writes in.

**Request body:**

    {
      "message": "What does this result mean?",
      "prediction_context": {
        "prediction": 1,
        "probability": 0.72,
        "diagnosis": "Diabetes detected"
      }
    }

---

## Development

### Prediction
The `/predict` endpoint loads `best_model.pkl` and `scaler.pkl` from the `ml/`
directory at startup — not on each request. This avoids reloading the model on
every call, keeping response times low.

Input data is scaled using the same `StandardScaler` fitted during training to
prevent data leakage. The model returns the predicted class and the probability
of the positive class (diabetes).

### Chat
The `/chat` endpoint uses Llama 3.1 8B via the Groq API. If a prediction context
is provided, it is prepended to the user message so the model can give a
contextually relevant explanation.

The system prompt instructs the model to act as a medical assistant, provide
empathetic explanations and always recommend consulting a doctor.

### Data validation
All input and output data is validated using Pydantic models. Invalid requests
(negative values, wrong types) are rejected before reaching the prediction logic.

---

## Usage

**Requirements:**
- Docker and Docker Compose
- A Groq API key (free at [console.groq.com](https://console.groq.com))
- The `best_model.pkl` and `scaler.pkl` files from [diabetes-classifier](https://github.com/Alro4503/diabetes-classifier)

**Setup:**

    git clone https://github.com/Alro4503/diabetes-api
    cd diabetes-api

Copy `best_model.pkl` and `scaler.pkl` from `diabetes-classifier/models/` to `ml/`.

Create a `.env` file in the root:

    GROQ_API_KEY=your_api_key_here

**Run with Docker:**

    docker-compose up --build

**Run locally:**

    pip install -r requirements.txt
    uvicorn app.main:app --reload

API docs available at `http://localhost:8000/docs`

---

## Project Structure

    diabetes-api/
    ├── app/
    │   ├── __init__.py
    │   ├── main.py         ← API entry point and endpoints
    │   ├── models.py       ← Pydantic input/output models
    │   ├── predict.py      ← prediction logic
    │   └── chat.py         ← Groq LLM integration
    ├── ml/
    │   └── .gitkeep        ← place best_model.pkl and scaler.pkl here
    ├── .env                ← API keys (not committed)
    ├── Dockerfile
    ├── docker-compose.yml
    └── requirements.txt