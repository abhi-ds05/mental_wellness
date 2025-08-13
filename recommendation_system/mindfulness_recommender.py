import os
import sys
import pandas as pd
from collections import Counter
from datetime import datetime

# ==== CONFIG ====
DATA_PATH = os.path.join("datasets", "synthetic_user_journals", "journal_entries_cleaned.csv")
OUTPUT_DIR = os.path.join("recommendations_system", "mindfulness_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== MINDFULNESS EXERCISES CATALOG ====
MINDFULNESS_EXERCISES = {
    "joy": [
        "Gratitude meditation: Focus on things you are grateful for today.",
        "Loving-kindness meditation to share your positive feelings."
    ],
    "sadness": [
        "Body scan meditation to connect with your physical sensations.",
        "Mindful breathing to gently acknowledge your feelings."
    ],
    "anger": [
        "4-7-8 deep breathing exercise to calm agitation.",
        "Mindfulness of emotions: Observe anger without judgment."
    ],
    "fear": [
        "Grounding exercise: Use 5-4-3-2-1 technique focusing on the present.",
        "Gentle mindful walking to release anxiety."
    ],
    "neutral": [
        "Mindful pause: take 3 minutes focusing on your breath.",
        "Observe your surroundings without judgment for 5 minutes."
    ],
    "apathy": [
        "Guided visualization to imagine energizing scenarios.",
        "Mindful movement: stretching or yoga with attention to sensations."
    ],
    "surprise": [
        "Open awareness meditation to explore new experiences without resistance.",
        "Practice acceptance with mindful reflection on unexpected feelings."
    ],
    "disgust": [
        "Mindfulness of senses: Engage with pleasant sensory experiences.",
        "Self-compassion meditation to counter negative reactions."
    ]
}

# ==== LOAD USER MOOD AND EMOTION HISTORY ====
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

# ==== SUGGEST MINDFULNESS EXERCISES BASED ON EMOTIONS ====
def suggest_mindfulness(emotion_list):
    emotion_counter = Counter([e for e in emotion_list if e])
    if not emotion_counter:
        return ["Try a brief mindful breathing exercise for 3 minutes to check in with yourself."]

    top_emotions = [emo for emo, _ in emotion_counter.most_common(2)]
    exercises_output = []

    for emo in top_emotions:
        exercises = MINDFULNESS_EXERCISES.get(emo, MINDFULNESS_EXERCISES.get("neutral"))
        exercises_output.append({
            "emotion": emo,
            "exercises": exercises
        })
    return exercises_output

# ==== SAVE EXERCISES TO CSV ====
def save_mindfulness_exercises(user_id, exercises):
    file_path = os.path.join(OUTPUT_DIR, f"mindfulness_exercises_{user_id}.csv")
    rows = []
    for item in exercises:
        for exercise in item["exercises"]:
            rows.append({
                "user_id": user_id,
                "emotion": item["emotion"],
                "exercise": exercise
            })
    pd.DataFrame(rows).to_csv(file_path, index=False)
    print(f"[INFO] Saved mindfulness exercises to {file_path}")

# ==== MAIN FUNCTION ====
def main(user_id):
    emotions, scores, _ = load_user_history(user_id)
    exercises = suggest_mindfulness(emotions)

    if not exercises:
        print(f"No mindfulness exercises could be generated for user {user_id}.")
        return

    print(f"\nMindfulness exercise suggestions for user {user_id} based on recent emotions {list(set(emotions))}:\n")
    for item in exercises:
        print(f"For emotion '{item['emotion']}':")
        for ex in item["exercises"]:
            print(f"  • {ex}")
        print()

    save_mindfulness_exercises(user_id, exercises)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate mindfulness recommendations for a user.")
    parser.add_argument('user_id', type=str, help='User ID to generate mindfulness recommendations for')
    args = parser.parse_args()
    main(args.user_id)
