import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==== Add root path ====
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ==== Import your existing recommendation modules ====
from recommendation_system.suggest_coping_stratergies import load_user_history, suggest_strategies
from recommendation_system.mindfulness_recommender import suggest_mindfulness
from adaptive_response_generation.gpt_adapter import generate_empathetic_message

# ==== FastAPI app ====
app = FastAPI(
    title="Mental Wellness AI FastAPI Server",
    description="Unified backend for chatbot, coping strategies, mindfulness, and empathetic responses",
    version="1.0.0"
)

# Allow all origins (change in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, allows all domains. Restrict in prod!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Models for requests ====
class ChatRequest(BaseModel):
    user_id: str
    message: str

class UserIDRequest(BaseModel):
    user_id: str

class EmpathyRequest(BaseModel):
    user_id: str
    top_emotion: str
    mood_trend: str = None
    tone_category: str = None
    recent_journal: str = None
    strategies: list = []
    mindfulness: list = []

# ==== Routes ====
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Mental Wellness AI API is running"}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    # Load user mood/emotion history
    recent_emotions, mood_scores, _ = load_user_history(req.user_id)
    top_emotion = recent_emotions[-1] if recent_emotions else "neutral"
    tone_category = "positive" if top_emotion in ["joy", "surprise"] else "neutral" if top_emotion == "neutral" else "negative"

    # Determine mood trend
    trend = None
    if mood_scores:
        if mood_scores[-1] > mood_scores[0]:
            trend = "improving"
        elif mood_scores[-1] < mood_scores[0]:
            trend = "declining"
        else:
            trend = "stable"

    # Get strategies and mindfulness suggestions
    coping_list = []
    for s in suggest_strategies(recent_emotions, mood_scores):
        coping_list.extend(s.get("quick", []))
    mindfulness_list = []
    for m in suggest_mindfulness(recent_emotions):
        mindfulness_list.extend(m.get("exercises", []))

    # Build GPT context & generate empathetic reply
    context = {
        "user_id": req.user_id,
        "top_emotion": top_emotion,
        "mood_trend": trend,
        "tone_category": tone_category,
        "recent_journal": req.message
    }
    ai_message = generate_empathetic_message(context, coping_list[:2], mindfulness_list[:2])

    return {
        "user_id": req.user_id,
        "top_emotion": top_emotion,
        "mood_trend": trend,
        "tone_category": tone_category,
        "ai_message": ai_message,
        "coping_suggestions": coping_list[:3],
        "mindfulness_suggestions": mindfulness_list[:3]
    }

@app.post("/coping")
def coping_endpoint(req: UserIDRequest):
    emotions, scores, _ = load_user_history(req.user_id)
    strategies = suggest_strategies(emotions, scores)
    return {"user_id": req.user_id, "strategies": strategies}

@app.post("/mindfulness")
def mindfulness_endpoint(req: UserIDRequest):
    emotions, _, _ = load_user_history(req.user_id)
    mindfulness = suggest_mindfulness(emotions)
    return {"user_id": req.user_id, "mindfulness": mindfulness}

@app.post("/empathetic")
def empathetic_endpoint(req: EmpathyRequest):
    context = {
        "user_id": req.user_id,
        "top_emotion": req.top_emotion,
        "mood_trend": req.mood_trend,
        "tone_category": req.tone_category,
        "recent_journal": req.recent_journal
    }
    message = generate_empathetic_message(context, req.strategies, req.mindfulness)
    return {"user_id": req.user_id, "empathetic_message": message}

# ==== Run the server ====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=5000, reload=True)
