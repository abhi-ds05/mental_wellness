import os
import sys
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import ast

# -------------------
# CONFIGURATION
# -------------------
DATA_PATH = os.path.join("datasets", "synthetic_user_journals", "journal_entries_cleaned.csv")
SUMMARY_OUTPUT_DIR = os.path.join("journaling_analysis", "summary_outputs")
PLOT_PATH = os.path.join("journaling_analysis", "sentiment_trend_plot.png")
os.makedirs(SUMMARY_OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"  # Replace with your model if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.3  # Multi-label classification threshold

# -------------------
# 1. LOAD MODEL
# -------------------
print("Loading BERT/DistilBERT model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

# -------------------
# 2. PREDICT EMOTIONS
# -------------------
def predict_emotions(texts):
    predictions = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        labels = [model.config.id2label[i] for i, p in enumerate(probs) if p >= THRESHOLD]
        predictions.append(labels)
    return predictions

# -------------------
# 3. LOAD DATA & ADD PREDICTIONS
# -------------------
print("Loading journal data...")
df = pd.read_csv(DATA_PATH)
time_col = 'timestamp' if 'timestamp' in df.columns else 'date'
df[time_col] = pd.to_datetime(df[time_col])

print("Predicting emotions for journal entries...")
df['predicted_emotions'] = predict_emotions(df['entry_text'].fillna(""))

# Save dataset with predictions for future use
predictions_path = os.path.join(SUMMARY_OUTPUT_DIR, "journal_entries_with_emotions.csv")
df.to_csv(predictions_path, index=False)

print("\nSample predictions:")
print(df[['user_id', 'predicted_emotions']].head())

# -------------------
# 4. GENERATE SUMMARIES
# -------------------
def overall_emotion_distribution(df, emotion_labels):
    all_emotions = [emo for sublist in df['predicted_emotions'] for emo in sublist]
    counts = Counter(all_emotions)
    total = sum(counts.values())
    distribution = {emotion: counts.get(emotion, 0) / total for emotion in emotion_labels}
    distribution_df = pd.DataFrame.from_dict(distribution, orient='index', columns=['proportion'])
    distribution_df = distribution_df.sort_values(by='proportion', ascending=False)
    return distribution_df

def per_user_top_emotions(df, user_col='user_id'):
    user_emotions = df.groupby(user_col)['predicted_emotions'].apply(lambda lists: [emo for sublist in lists for emo in sublist])
    user_top3 = user_emotions.apply(lambda emos: [emotion for emotion, _ in Counter(emos).most_common(3)])
    return user_top3

def daily_emotion_presence(df, time_col, emotion_labels):
    for emo in emotion_labels:
        df[emo] = df['predicted_emotions'].apply(lambda lst: 1 if emo in lst else 0)
    daily_avg = df.groupby(df[time_col].dt.date)[emotion_labels].mean()
    return daily_avg

def daily_dominant_emotion(daily_avg):
    dominant = daily_avg.idxmax(axis=1)
    dominant_counts = daily_avg.max(axis=1)
    summary_df = pd.DataFrame({'dominant_emotion': dominant, 'presence': dominant_counts})
    return summary_df

def example_entries_per_emotion(df, emotion_labels, text_col='entry_text', max_examples=3):
    examples = {}
    for emo in emotion_labels:
        filtered = df[df['predicted_emotions'].apply(lambda lst: emo in lst)]
        examples[emo] = filtered[text_col].dropna().head(max_examples).tolist()
    return examples

print("\nDetermining all emotion labels...")
unique_emotions = sorted(set(e for sublist in df['predicted_emotions'] for e in sublist))

print("\nGenerating overall emotion distribution...")
overall_dist = overall_emotion_distribution(df, unique_emotions)
overall_dist.to_csv(os.path.join(SUMMARY_OUTPUT_DIR, 'overall_emotion_distribution.csv'))
print(overall_dist)

print("\nGenerating top 3 emotions per user...")
user_top_emotions = per_user_top_emotions(df)
user_top_emotions.to_csv(os.path.join(SUMMARY_OUTPUT_DIR, 'per_user_top3_emotions.csv'), header=['top_emotions'])
print(user_top_emotions.head())

print("\nCalculating daily average emotion presence...")
daily_avg = daily_emotion_presence(df, time_col, unique_emotions)
daily_avg.to_csv(os.path.join(SUMMARY_OUTPUT_DIR, 'daily_emotion_presence.csv'))
print(daily_avg.head())

print("\nIdentifying dominant emotion per day...")
daily_dom = daily_dominant_emotion(daily_avg)
daily_dom.to_csv(os.path.join(SUMMARY_OUTPUT_DIR, 'daily_dominant_emotion.csv'))
print(daily_dom.head())

print("\nExtracting example journal entries per emotion...")
examples = example_entries_per_emotion(df, unique_emotions)
for emotion, texts in examples.items():
    safe_emo = emotion.replace(' ', '_').replace('/', '_')
    with open(os.path.join(SUMMARY_OUTPUT_DIR, f'examples_{safe_emo}.txt'), 'w', encoding='utf-8') as f:
        for entry in texts:
            f.write(entry + "\n\n")

# -------------------
# 5. PLOT TREND
# -------------------
print("\nPlotting sentiment trend...")
plt.figure(figsize=(12, 6))
for emo in unique_emotions:
    plt.plot(daily_avg.index, daily_avg[emo], label=emo)
plt.legend()
plt.title("Daily Average Emotion Presence")
plt.xlabel("Date")
plt.ylabel("Average Presence (0-1)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOT_PATH)

print(f"\n✅ Summary outputs saved in: {SUMMARY_OUTPUT_DIR}")
print(f"✅ Sentiment trend plot saved at: {PLOT_PATH}")
