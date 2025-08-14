from fastapi import APIRouter
from pydantic import BaseModel

# Import your existing backend logic
from recommendation_system.suggest_coping_stratergies import load_user_history, suggest_strategies
from recommendation_system.mindfulness_recommender import suggest_mindfulness
from adaptive_response_generation.gpt_adapter import generate_empathetic_message

# Create router instance
router = APIRouter()

# Request model
class ChatRequest(BaseModel):
    user_id: str
    message: str

@router.post("/chat")
def chat_endpoint(req: ChatRequest):
    """
    Main chat endpoint:
    - Loads the user's recent emotions/mood history.
    - Determines top emotion, tone, and mood trend.
    - Gets coping + mindfulness recommendations.
    - Uses GPT adapter to generate a warm empathetic reply.
    """

    # ==== Load user state ====
    emotions, scores, _ = load_user_history(req.user_id)

    top_emotion = emotions[-1] if emotions else "neutral"
    tone_category = (
        "positive" if top_emotion in ["joy", "surprise"]
        else "neutral" if top_emotion == "neutral"
        else "negative"
    )

    # ==== Mood trend detection ====
    trend = None
    if scores:
        if scores[-1] > scores[0]:
            trend = "improving"
        elif scores[-1] < scores[0]:
            trend = "declining"
        else:
            trend = "stable"

    # ==== Recommendations ====
    coping_suggestions = []
    for s in suggest_strategies(emotions, scores):
        coping_suggestions.extend(s.get("quick", []))

    mindfulness_suggestions = []
    for m in suggest_mindfulness(emotions):
        mindfulness_suggestions.extend(m.get("exercises", []))

    # ==== Build GPT context ====
    context = {
        "user_id": req.user_id,
        "top_emotion": top_emotion,
        "mood_trend": trend,
        "tone_category": tone_category,
        "recent_journal": req.message
    }

    # ==== Generate empathetic AI response ====
    ai_message = generate_empathetic_message(
        context,
        coping_suggestions[:2],
        mindfulness_suggestions[:2]
    )

    # ==== Return JSON response ====
    return {
        "user_id": req.user_id,
        "top_emotion": top_emotion,
        "mood_trend": trend,
        "tone_category": tone_category,
        "ai_message": ai_message,
        "coping_suggestions": coping_suggestions[:3],
        "mindfulness_suggestions": mindfulness_suggestions[:3]
    }
