import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import joblib
import nltk

# Add data_processing folder to sys.path for importing clean_text.py
data_processing_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
print(f"Adding to Python path for import: {data_processing_path}")
sys.path.append(data_processing_path)

try:
    from clean_text import clean_text
    print("Successfully imported clean_text.")
except ImportError as e:
    print(f"ImportError: {e}")
    print("Current sys.path:")
    for p in sys.path:
        print(f"  {p}")
    raise

# Download nltk resources if missing
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No dataset found at {csv_path}")
    print(f"Loading data from: {os.path.abspath(csv_path)}")
    df = pd.read_csv(csv_path)
    return df

def preprocess_texts(texts):
    return texts.apply(lambda t: clean_text(str(t)))

def prepare_labels(df):
    exclude_cols = ['text', 'comment', 'id', 'subreddit', 'created_utc', 'parent_id']
    # Select columns that are not excluded and have binary-like values (0,1)
    emotion_cols = [
        col for col in df.columns
        if col not in exclude_cols and df[col].nunique() <= 2 and
           pd.api.types.is_numeric_dtype(df[col])
    ]
    y = df[emotion_cols].fillna(0).astype(int)

    # Drop emotion columns with no positive examples
    empty_cols = [col for col in emotion_cols if y[col].sum() == 0]
    if empty_cols:
        print(f"Dropping empty emotion columns with no positive samples: {empty_cols}")
        y = y.drop(columns=empty_cols)
        emotion_cols = [col for col in emotion_cols if col not in empty_cols]

    print(f"Final emotion label columns used: {emotion_cols}")
    return y, emotion_cols

def main():
    # Correct relative path assuming run from the project root
    data_path = os.path.join('datasets', 'GOEMOTIONS', 'goemotions_full.csv')

    # Load dataset
    df = load_data(data_path)

    # Determine text column name
    text_column = 'text' if 'text' in df.columns else 'comment'
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Preprocess text
    print("Cleaning texts...")
    df[text_column] = preprocess_texts(df[text_column])
    X = df[text_column]

    # Prepare labels
    y, emotion_labels = prepare_labels(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Vectorize text
    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=8000, min_df=2, max_df=0.9)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train classifier
    print("Training multi-label Logistic Regression classifier...")
    clf = OneVsRestClassifier(LogisticRegression(max_iter=300, class_weight='balanced'))
    clf.fit(X_train_vec, y_train)

    # Evaluate model
    print("Evaluating model on test data...")
    y_pred = clf.predict(X_test_vec)
    report = classification_report(y_test, y_pred, target_names=emotion_labels, zero_division=0)
    print(report)

    # Save model, vectorizer, and emotion labels
    model_dir = os.path.join('emotion_classification', 'models')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, 'text_emotion_classifier.joblib'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.joblib'))
    joblib.dump(emotion_labels, os.path.join(model_dir, 'emotion_labels.joblib'))
    print(f"Model, vectorizer, and emotion labels saved in: {model_dir}")

    # Sample predictions from test set
    print("\nSample predictions on test set:")
    sample_texts = X_test.sample(5, random_state=42)
    for text in sample_texts:
        vec = vectorizer.transform([text])
        pred = clf.predict(vec)
        emotions = [emotion_labels[i] for i, val in enumerate(pred[0]) if val == 1]
        print(f"Text: {text}")
        print(f"Predicted emotions: {emotions}\n")

if __name__ == "__main__":
    main()
