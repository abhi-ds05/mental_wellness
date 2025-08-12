import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# ==== CONFIG ====
DATA_PATH = os.path.join("datasets", "synthetic_user_journals", "journal_entries_cleaned.csv")
OUTPUT_DIR = os.path.join("journaling_analysis", "mood_history_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== DATA_PROCESSING PATH ====
data_processing_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
if data_processing_path not in sys.path:
    sys.path.insert(0, data_processing_path)

from clean_text import clean_text

# ==== LOAD & PREPARE DATA ====
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Journal data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    time_col = "timestamp" if "timestamp" in df.columns else "date"
    df[time_col] = pd.to_datetime(df[time_col])

    # If predicted_emotions not available, fallback to single 'emotion' column if it exists
    if "predicted_emotions" not in df.columns:
        if "emotion" in df.columns:
            df["predicted_emotions"] = df["emotion"].apply(lambda e: [e] if pd.notna(e) else [])
        else:
            raise ValueError("No 'predicted_emotions' or 'emotion' column found in dataset.")

    # Parse if stored as string
    if isinstance(df["predicted_emotions"].iloc[0], str):
        import ast
        df["predicted_emotions"] = df["predicted_emotions"].apply(ast.literal_eval)

    return df, time_col

# ==== EMOTION PRESENCE ENCODING ====
def one_hot_encode_emotions(df, emotion_labels):
    for emo in emotion_labels:
        df[emo] = df["predicted_emotions"].apply(lambda lst: 1 if emo in lst else 0)
    return df

# ==== AGGREGATION FUNCTIONS ====
def aggregate_mood(df, time_col, emotion_labels, freq='D'):
    grouped = df.groupby([pd.Grouper(key=time_col, freq=freq)])[emotion_labels].mean()
    return grouped

def per_user_mood_history(df, time_col, emotion_labels, freq='D'):
    grouped = df.groupby(['user_id', pd.Grouper(key=time_col, freq=freq)])[emotion_labels].mean()
    return grouped

# ==== INTERACTIVE PLOTTING ====
def plot_overall_mood_interactive(history_df, emotion_labels, freq_label):
    fig = go.Figure()
    for emo in emotion_labels:
        fig.add_trace(go.Scatter(
            x=history_df.index,
            y=history_df[emo],
            mode='lines+markers',
            name=emo,
            hovertemplate='%{x}<br>' + emo + ': %{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title=f"Overall Mood History ({freq_label})",
        xaxis_title='Date',
        yaxis_title='Average Presence (0-1)',
        legend_title='Emotions',
        hovermode='x unified'
    )

    output_file = os.path.join(OUTPUT_DIR, f"interactive_overall_mood_{freq_label}.html")
    fig.write_html(output_file)
    print(f"[INFO] Saved interactive plot: {output_file}")

def plot_user_mood_interactive(history_df, emotion_labels, user_id, freq_label):
    if user_id not in history_df.index:
        print(f"[WARN] User ID {user_id} not found in history data.")
        return

    user_df = history_df.loc[user_id]
    fig = go.Figure()
    for emo in emotion_labels:
        fig.add_trace(go.Scatter(
            x=user_df.index,
            y=user_df[emo],
            mode='lines+markers',
            name=emo,
            hovertemplate='%{x}<br>' + emo + ': %{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title=f"Mood History for {user_id} ({freq_label})",
        xaxis_title='Date',
        yaxis_title='Average Presence (0-1)',
        legend_title='Emotions',
        hovermode='x unified'
    )

    output_file = os.path.join(OUTPUT_DIR, f"interactive_mood_{user_id}_{freq_label}.html")
    fig.write_html(output_file)
    print(f"[INFO] Saved interactive plot: {output_file}")

# ==== MAIN FUNCTION ====
def generate_mood_history_interactive():
    print("[INFO] Loading data...")
    df, time_col = load_data()

    print("[INFO] Determining emotion labels...")
    emotion_labels = sorted(set(e for sublist in df["predicted_emotions"] for e in sublist))
    print("[INFO] Emotion labels found:", emotion_labels)

    print("[INFO] One-hot encoding emotions...")
    df = one_hot_encode_emotions(df, emotion_labels)

    print("\n[INFO] Aggregating overall mood history...")
    daily_history = aggregate_mood(df, time_col, emotion_labels, freq='D')
    weekly_history = aggregate_mood(df, time_col, emotion_labels, freq='W')
    monthly_history = aggregate_mood(df, time_col, emotion_labels, freq='M')

    daily_history.to_csv(os.path.join(OUTPUT_DIR, "overall_mood_daily.csv"))
    weekly_history.to_csv(os.path.join(OUTPUT_DIR, "overall_mood_weekly.csv"))
    monthly_history.to_csv(os.path.join(OUTPUT_DIR, "overall_mood_monthly.csv"))

    plot_overall_mood_interactive(daily_history, emotion_labels, "daily")
    plot_overall_mood_interactive(weekly_history, emotion_labels, "weekly")
    plot_overall_mood_interactive(monthly_history, emotion_labels, "monthly")

    print("\n[INFO] Aggregating per-user mood history...")
    daily_user_history = per_user_mood_history(df, time_col, emotion_labels, freq='D')
    weekly_user_history = per_user_mood_history(df, time_col, emotion_labels, freq='W')

    daily_user_history.to_csv(os.path.join(OUTPUT_DIR, "per_user_mood_daily.csv"))
    weekly_user_history.to_csv(os.path.join(OUTPUT_DIR, "per_user_mood_weekly.csv"))

    # Interactive plot for first 3 users as example
    unique_users = df["user_id"].unique()
    for user_id in unique_users[:3]:
        plot_user_mood_interactive(daily_user_history, emotion_labels, user_id, "daily")
        plot_user_mood_interactive(weekly_user_history, emotion_labels, user_id, "weekly")

    print(f"\n✅ Interactive mood history generation complete. Outputs saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_mood_history_interactive()
