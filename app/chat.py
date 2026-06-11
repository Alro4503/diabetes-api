import os
from groq import Groq
from dotenv import load_dotenv
from app.models import ChatMessage, ChatResponse

# --- Load environment variables ---
load_dotenv()

# --- Init Groq client ---
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- System prompt ---
SYSTEM_PROMPT = """You are a medical assistant specialized in diabetes analysis. 
You help patients and medical staff understand diabetes prediction results.
You provide clear, accurate and empathetic explanations.
Always recommend consulting a doctor for medical decisions.
Answer in the same language the user writes in."""

# --- Chat ---
def chat(message: ChatMessage) -> ChatResponse:
    context = ""
    if message.prediction_context:
        context = f"""
Current patient prediction:
- Diagnosis: {message.prediction_context.diagnosis}
- Diabetes probability: {message.prediction_context.probability * 100:.1f}%
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\nUser question: {message.message}"}
        ]
    )

    return ChatResponse(response=response.choices[0].message.content)