from fastapi import APIRouter
from pydantic import BaseModel

# Try safe imports
try:
    from journaling_analysis.sentiment_trend_tracker import get_mood_trend
except ImportError:
    get_mood_trend = None

try:
    from emotion_classification.text_emotion_classifier import predict_emotion
except ImportError:
    predict_emotion = None

router = APIRouter()

# ====== Request Models ======
class MoodRequest(BaseModel):
    user_id: str

class TextEmotionRequest(BaseModel):
    text: str

# ====== Endpoints ======

@router.post("/mood/trend")
def mood_trend(req: MoodRequest):
    """
    Returns the user's mood trend (improving, declining, stable) based on recent history.
    """
    if not get_mood_trend:
        return {"error": "get_mood_trend function not available in journaling_analysis.sentiment_trend_tracker"}

    try:
        trend = get_mood_trend(req.user_id)
    except Exception as e:
        return {"error": str(e)}

    return {"user_id": req.user_id, "mood_trend": trend}


@router.post("/mood/text_emotion")
def text_emotion(req: TextEmotionRequest):
    """
    Predicts the emotion from a given text snippet.
    """
    if not predict_emotion:
        return {"error": "predict_emotion function not available in emotion_classification.text_emotion_classifier"}

    try:
        emotion = predict_emotion(req.text)
    except Exception as e:
        return {"error": str(e)}

    return {"text": req.text, "predicted_emotion": emotion}
