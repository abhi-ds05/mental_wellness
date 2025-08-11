import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

# ==== CONFIG ====
DATA_PATH = os.path.join("datasets", "synthetic_user_journals", "journal_entries_cleaned.csv")
MODEL_DIR = os.path.join("emotion_classification", "models")

BERT_MODEL_FILE = "distilbert_emotion_model.bin"
IS_BERT_MODEL = True  # True for BERT/DistilBERT, False for TF-IDF

TOP_N_EMOTIONS = 3  # keep top 3 emotions per entry

# ==== DATA_PROCESSING PATH ====
data_processing_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
if data_processing_path not in sys.path:
    sys.path.insert(0, data_processing_path)

from clean_text import clean_text

# ==== LOADERS ====
def load_tfidf_model():
    clf = joblib.load(os.path.join(MODEL_DIR, "text_emotion_classifier.joblib"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    labels = joblib.load(os.path.join(MODEL_DIR, "emotion_labels.joblib"))
    return clf, vectorizer, labels

def load_bert_model():
    import torch
    from emotion_classification.emotion_model import DistilBertForMultiLabelClassification

    checkpoint = torch.load(
        os.path.join(MODEL_DIR, BERT_MODEL_FILE),
        map_location='cpu',
        weights_only=False
    )
    labels = checkpoint['emotion_labels']
    tokenizer = checkpoint['tokenizer']

    model = DistilBertForMultiLabelClassification(len(labels))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, tokenizer, labels

# ==== PREDICTORS ====
def predict_emotions_tfidf(text, clf, vectorizer, labels):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = clf.predict(vec)[0]
    return [labels[i] for i, val in enumerate(pred) if val == 1]

def predict_emotions_bert(text, model, tokenizer, labels, top_n=TOP_N_EMOTIONS):
    import torch
    cleaned = clean_text(text)
    encoding = tokenizer.encode_plus(
        cleaned,
        add_special_tokens=True,
        truncation=True,
        max_length=128,
        padding='max_length',
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(encoding['input_ids'], encoding['attention_mask'])
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    # Get indices sorted by probability
    sorted_indices = probs.argsort()[::-1]
    selected_emotions = []

    # Always pick top N
    for idx in sorted_indices[:top_n]:
        if labels[idx] == "neutral":
            # Only include neutral if it's the top probability
            if idx == sorted_indices[0]:
                selected_emotions.append("neutral")
        else:
            selected_emotions.append(labels[idx])

    return selected_emotions

# ==== TREND TRACKING ====
def track_emotion_trends():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Journal data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    text_column = 'entry_text' if 'entry_text' in df.columns else 'text'
    time_column = 'timestamp' if 'timestamp' in df.columns else 'date'
    df[time_column] = pd.to_datetime(df[time_column])

    if IS_BERT_MODEL:
        print("Loading BERT/DistilBERT model...")
        model, tokenizer, labels = load_bert_model()
        predictor = lambda text: predict_emotions_bert(text, model, tokenizer, labels)
    else:
        print("Loading TF-IDF model...")
        clf, vectorizer, labels = load_tfidf_model()
        predictor = lambda text: predict_emotions_tfidf(text, clf, vectorizer, labels)

    print("Predicting emotions for journal entries...")
    df['predicted_emotions'] = df[text_column].apply(lambda x: predictor(str(x)))

    # One-hot encode predictions
    for label in labels:
        df[label] = df['predicted_emotions'].apply(lambda emo_list: 1 if label in emo_list else 0)

    # DAILY aggregation
    df_daily = df.groupby(df[time_column].dt.date)[labels].mean()

    # Plot daily trends
    plt.figure(figsize=(14, 8))
    for label in labels:
        plt.plot(df_daily.index, df_daily[label], label=label, alpha=0.6)
    plt.title("Emotion Trends Over Time from Journals")
    plt.xlabel("Date")
    plt.ylabel("Average Presence (0-1)")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    plt.grid(True)
    plt.tight_layout()

    output_plot = os.path.join("journaling_analysis", "sentiment_trend_plot.png")
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot, dpi=300)
    print(f"Saved plot to {output_plot}")

    return df, df_daily

# ==== ENTRY POINT ====
if __name__ == "__main__":
    full_df, trend_df = track_emotion_trends()
    print("\nSample predictions:")
    print(full_df[[full_df.columns[0], 'predicted_emotions']].head())
