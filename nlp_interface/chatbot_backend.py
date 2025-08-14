import os
import sys
from flask import Flask, request, jsonify

# ===== Add project root to path =====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ===== Import your existing modules =====
from recommendation_system.suggest_coping_stratergies import load_user_history, suggest_strategies
from recommendation_system.mindfulness_recommender import suggest_mindfulness
from adaptive_response_generation.gpt_adapter import generate_empathetic_message

app = Flask(__name__)

# ==== A simple utility for cleaning incoming text ====
def clean_user_input(user_text):
    return user_text.strip()

# ==== Process user message ====
def process_message(user_id, message_text):
    """
    Main logic: takes a user message, analyzes mood/emotions,
    generates an empathetic response using GPT adapter.
    """
    cleaned_text = clean_user_input(message_text)

    # Get recent mood/emotion history
    recent_emotions, mood_scores, user_df = load_user_history(user_id)

    # Fall back to neutral if no emotions found yet
    top_emotion = recent_emotions[-1] if recent_emotions else "neutral"
    tone_category = "positive" if top_emotion in ["joy", "surprise"] \
                    else "neutral" if top_emotion == "neutral" \
                    else "negative"

    # Detect mood trend
    trend = None
    if mood_scores:
        if mood_scores[-1] > mood_scores[0]:
            trend = "improving"
        elif mood_scores[-1] < mood_scores[0]:
            trend = "declining"
        else:
            trend = "stable"

    # Get coping strategies & mindfulness ideas
    coping_suggestions = suggest_strategies(recent_emotions, mood_scores)
    coping_list = []
    for s in coping_suggestions:
        coping_list.extend(s.get("quick", []))
    mindfulness_suggestions = suggest_mindfulness(recent_emotions)
    mindfulness_list = []
    for m in mindfulness_suggestions:
        mindfulness_list.extend(m.get("exercises", []))

    # Prepare context for GPT adapter
    user_context = {
        "user_id": user_id,
        "top_emotion": top_emotion,
        "mood_trend": trend,
        "tone_category": tone_category,
        "recent_journal": cleaned_text
    }

    # Generate AI-powered empathetic message
    ai_message = generate_empathetic_message(
        user_context,
        strategies=coping_list[:2],         # Limit to 2 suggestions
        mindfulness=mindfulness_list[:2]    # Limit to 2 suggestions
    )

    return {
        "user_id": user_id,
        "top_emotion": top_emotion,
        "mood_trend": trend,
        "tone_category": tone_category,
        "ai_message": ai_message,
        "coping_suggestions": coping_list[:3],
        "mindfulness_suggestions": mindfulness_list[:3]
    }

# ==== API routes ====

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """
    POST JSON format:
    {
        "user_id": "u001",
        "message": "I feel stressed about work lately."
    }
    """
    data = request.get_json()
    if not data or "user_id" not in data or "message" not in data:
        return jsonify({"error": "user_id and message are required"}), 400

    user_id = data["user_id"]
    message_text = data["message"]

    try:
        response_data = process_message(user_id, message_text)
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Chatbot backend running"})


if __name__ == "__main__":
    # Run backend for local testing
    app.run(host="0.0.0.0", port=5000, debug=True)
