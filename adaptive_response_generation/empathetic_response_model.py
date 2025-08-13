import os
import pandas as pd
from collections import Counter
from datetime import datetime

# Reuse functions from coping & mindfulness modules
from recommendation_system.suggest_coping_stratergies import load_user_history, suggest_strategies
from recommendation_system.mindfulness_recommender import suggest_mindfulness

OUTPUT_DIR = os.path.join("recommendations_system_output", "empathetic_responses")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== Tone templates for empathy ====
EMPATHY_TEMPLATES = {
    "positive": [
        "I'm so happy to hear you're feeling good. Let's celebrate and keep the energy flowing.",
        "It's uplifting to see joy in your recent mood. Cherish these moments and spread the positivity."
    ],
    "neutral": [
        "Thanks for sharing how you're feeling. Let's keep things balanced and look for small ways to brighten your day.",
        "I appreciate you checking in. Even steady moods deserve care and gentle attention."
    ],
    "negative": [
        "I can feel this has been tough for you. I’m here to listen and support, one step at a time.",
        "It sounds like you’re carrying a lot right now. That’s okay — together, we can find small moments of relief."
    ]
}

# Map emotion to tone category
EMO_TO_TONE_CAT = {
    "joy": "positive",
    "surprise": "positive",
    "sadness": "negative",
    "fear": "negative",
    "anger": "negative",
    "apathy": "negative",
    "disgust": "negative",
    "neutral": "neutral"
}

def generate_empathetic_response(user_id):
    # ==== Load user’s history ====
    recent_emotions, mood_scores, _ = load_user_history(user_id)

    if recent_emotions:
        top_emotion = Counter(recent_emotions).most_common(1)[0][0]
    else:
        top_emotion = "neutral"

    tone_cat = EMO_TO_TONE_CAT.get(top_emotion, "neutral")

    # ==== Mood Trend Detection ====
    trend = None
    if mood_scores:
        if mood_scores[-1] > mood_scores[0]:
            trend = "improving"
        elif mood_scores[-1] < mood_scores[0]:
            trend = "declining"
        else:
            trend = "stable"

    # ==== Base Empathy Statement ====
    import random
    empathy_line = random.choice(EMPATHY_TEMPLATES[tone_cat])

    # ==== Add Relevant Coping Strategy ====
    strategies = suggest_strategies(recent_emotions, mood_scores)
    suggestion_line = None
    if strategies:
        for strat in strategies:
            if strat["emotion"] == top_emotion and strat["quick"]:
                suggestion_line = strat["quick"][0]
                break
    if not suggestion_line and strategies:
        suggestion_line = strategies[0]["quick"][0] if strategies[0]["quick"] else None

    # ==== Add Mindfulness Suggestion ====
    mindfulness_suggestions = suggest_mindfulness(recent_emotions)
    mindfulness_line = None
    if mindfulness_suggestions:
        for sug in mindfulness_suggestions:
            if sug["emotion"] == top_emotion:
                mindfulness_line = sug["exercises"][0]
                break
    if not mindfulness_line and mindfulness_suggestions:
        mindfulness_line = mindfulness_suggestions[0]["exercises"][0]

    # ==== Construct Final Empathetic Response ====
    final_parts = [empathy_line]
    if suggestion_line:
        final_parts.append(f"Here's something you could try: {suggestion_line}")
    if mindfulness_line:
        final_parts.append(f"Mindfulness idea: {mindfulness_line}")

    final_message = " ".join(final_parts)

    # ==== Save to CSV log ====
    out_path = os.path.join(OUTPUT_DIR, f"empathetic_response_{user_id}.csv")
    pd.DataFrame([{
        "user_id": user_id,
        "top_emotion": top_emotion,
        "tone_category": tone_cat,
        "mood_trend": trend,
        "response": final_message,
        "timestamp": datetime.now()
    }]).to_csv(out_path, index=False)

    # ==== Print to console ====
    print(f"\n[Empathetic Response for {user_id}]")
    print(final_message)
    print(f"[INFO] Saved to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate empathetic wellness responses for a user.")
    parser.add_argument("user_id", type=str, help="User ID to generate a response for")
    args = parser.parse_args()
    generate_empathetic_response(args.user_id)
