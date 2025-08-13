import os
import sys
import pandas as pd
from collections import Counter
from datetime import datetime

# ==== CONFIG ====
DATA_PATH = os.path.join("datasets", "synthetic_user_journals", "journal_entries_cleaned.csv")
OUTPUT_DIR = os.path.join("recommendations_system", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== STRATEGY LIBRARY ====
COPING_STRATEGIES = {
    "joy": {
        "quick": [
            "Share your joy with someone you care about.",
            "Write down three things you are grateful for today."
        ],
        "reflection": [
            "Reflect on what contributed to this joy and how to repeat it.",
            "Capture this happy moment in a photo or journal."
        ],
        "long_term": [
            "Engage in regular activities that bring you joy.",
            "Volunteer or help others to spread positivity."
        ]
    },
    "sadness": {
        "quick": [
            "Call or message a supportive friend.",
            "Step outside for a short walk in nature."
        ],
        "reflection": [
            "Write about what's making you feel sad and possible ways to cope.",
            "Listen to calming or uplifting music."
        ],
        "long_term": [
            "Maintain a routine that includes social and physical activity.",
            "Consider talking to a counselor if sadness persists."
        ]
    },
    "anger": {
        "quick": [
            "Practice 4-7-8 deep breathing.",
            "Walk away from the trigger for 10 minutes."
        ],
        "reflection": [
            "Write the reasons for your anger without judgement.",
            "Identify a constructive action to address the cause."
        ],
        "long_term": [
            "Regular mindfulness or yoga practice.",
            "Develop assertive communication skills."
        ]
    },
    "fear": {
        "quick": [
            "Ground yourself with the 5-4-3-2-1 technique.",
            "Focus on slow, steady breathing."
        ],
        "reflection": [
            "Challenge fearful thoughts with evidence.",
            "Talk about your fear with a trusted person."
        ],
        "long_term": [
            "Gradual exposure to feared situations.",
            "Track and celebrate small wins against your fears."
        ]
    },
    "neutral": {
        "quick": [
            "Check in on your goals for the day.",
            "Do a 5-minute body stretch."
        ],
        "reflection": [
            "Write 3 small things you could improve today.",
            "Evaluate your current priorities."
        ],
        "long_term": [
            "Cultivate hobbies to enrich daily life.",
            "Schedule regular wellbeing check-ins."
        ]
    }
}

# ==== LOAD USER HISTORY ====
def load_user_history(user_id):
    df = pd.read_csv(DATA_PATH)
    time_col = 'timestamp' if 'timestamp' in df.columns else 'date'
    df[time_col] = pd.to_datetime(df[time_col])

    if 'predicted_emotions' in df.columns:
        if isinstance(df['predicted_emotions'].iloc[0], str):
            import ast
            df['predicted_emotions'] = df['predicted_emotions'].apply(ast.literal_eval)
        user_df = df[df['user_id'] == user_id]
        recent_emotions = []
        for lst in user_df['predicted_emotions'].tail(7):
            recent_emotions.extend(lst)
    elif 'emotion' in df.columns:
        user_df = df[df['user_id'] == user_id]
        recent_emotions = list(user_df['emotion'].tail(7))
    else:
        raise ValueError("Emotion column ('emotion' or 'predicted_emotions') not found.")

    recent_scores = user_df['mood_score'].tail(7).tolist() if 'mood_score' in user_df.columns else []
    return recent_emotions, recent_scores, user_df

# ==== STRATEGY SUGGESTION ====
def suggest_strategies(emotion_list, mood_scores):
    emotion_counter = Counter([e for e in emotion_list if e])
    if not emotion_counter:
        return []

    top_emotions = [emo for emo, _ in emotion_counter.most_common(2)]
    strategies_output = []

    # Detect mood trend if scores are available
    trend = None
    if mood_scores:
        if mood_scores[-1] > mood_scores[0]:
            trend = "improving"
        elif mood_scores[-1] < mood_scores[0]:
            trend = "declining"
        else:
            trend = "stable"

    for emo in top_emotions:
        emo_strats = COPING_STRATEGIES.get(emo, COPING_STRATEGIES.get("neutral", {}))
        strategies_output.append({
            "emotion": emo,
            "trend": trend,
            "quick": emo_strats["quick"],
            "reflection": emo_strats["reflection"],
            "long_term": emo_strats["long_term"]
        })
    return strategies_output

# ==== SAVE RECOMMENDATIONS ====
def save_recommendations(user_id, strategies):
    rec_file = os.path.join(OUTPUT_DIR, f"recommendations_{user_id}.csv")
    rows = []
    for strat in strategies:
        for category in ["quick", "reflection", "long_term"]:
            for s in strat[category]:
                rows.append({
                    "user_id": user_id,
                    "emotion": strat["emotion"],
                    "trend": strat["trend"],
                    "category": category,
                    "strategy": s
                })
    pd.DataFrame(rows).to_csv(rec_file, index=False)
    print(f"[INFO] Saved recommendations to {rec_file}")

# ==== MAIN ====
def main(user_id):
    emotions, scores, user_df = load_user_history(user_id)
    strategies = suggest_strategies(emotions, scores)

    if not strategies:
        print(f"No strategies could be generated for {user_id}.")
        return
    
    print(f"\nRecent dominant emotions for {user_id}: {[s['emotion'] for s in strategies]}")
    if strategies[0]["trend"]:
        print(f"Mood trend detected: {strategies[0]['trend']}\n")

    for strat in strategies:
        print(f"--- For emotion '{strat['emotion']}' ---")
        for cat in ["quick", "reflection", "long_term"]:
            print(f"{cat.capitalize()} strategies:")
            for s in strat[cat]:
                print(f"  • {s}")
        print()

    save_recommendations(user_id, strategies)

if __name__ == "__main__":
    # Example test user
    main("u001")
