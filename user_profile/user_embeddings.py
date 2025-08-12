import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel

# Paths & config
DATA_PATH = os.path.join("datasets", "synthetic_user_journals", "journal_entries_cleaned.csv")
OUTPUT_DIR = os.path.join("user_profiles")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_MODEL_NAME = "distilbert-base-uncased"

# Add data_processing path for clean_text
data_processing_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
if data_processing_path not in sys.path:
    sys.path.insert(0, data_processing_path)
from clean_text import clean_text

# Load models (replace stubs with real models for audio/visual if available)
tokenizer = DistilBertTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
text_model = DistilBertModel.from_pretrained(EMBEDDING_MODEL_NAME)
text_model.to(DEVICE)
text_model.eval()

def extract_text_embedding(text):
    cleaned = clean_text(text)
    encoding = tokenizer(cleaned, add_special_tokens=True, truncation=True,
                        padding='max_length', max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = text_model(
            input_ids=encoding["input_ids"].to(DEVICE),
            attention_mask=encoding["attention_mask"].to(DEVICE)
        )
        hidden_state = outputs.last_hidden_state
        mask = encoding["attention_mask"].to(DEVICE).unsqueeze(-1).expand(hidden_state.size())
        masked_hidden = hidden_state * mask
        summed = torch.sum(masked_hidden, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counts
        return mean_pooled.cpu().numpy().flatten()

def extract_audio_embedding(audio_path):
    # TODO: Replace with real feature extraction (e.g., MFCCs, wav2vec)
    return np.zeros(32)

def extract_visual_embedding(image_path):
    # TODO: Replace with real feature extraction (e.g., CNN features)
    return np.zeros(64)

def get_user_modalities(group):
    # Example: 'audio_path' and 'image_path' columns expected/integrated in journaling CSV
    audio_embeds = []
    visual_embeds = []
    for _, row in group.iterrows():
        audio_path = row.get('audio_path')
        image_path = row.get('image_path')
        # Only aggregate if columns exist and path is valid
        if isinstance(audio_path, str) and audio_path.strip():
            audio_embeds.append(extract_audio_embedding(audio_path))
        if isinstance(image_path, str) and image_path.strip():
            visual_embeds.append(extract_visual_embedding(image_path))
    audio_vec = np.mean(audio_embeds, axis=0) if audio_embeds else np.zeros(32)
    visual_vec = np.mean(visual_embeds, axis=0) if visual_embeds else np.zeros(64)
    return audio_vec, visual_vec

def compute_user_embeddings(df, text_col="entry_text"):
    user_embeddings = {}
    for user_id, group in tqdm(df.groupby("user_id"), desc="Computing user multimodal embeddings"):
        # Aggregate text
        text_embeds = [extract_text_embedding(txt) for txt in group[text_col] if isinstance(txt, str) and txt.strip()]
        text_vec = np.mean(text_embeds, axis=0) if text_embeds else np.zeros(768)
        # Aggregate audio/visual
        audio_vec, visual_vec = get_user_modalities(group)
        # Multimodal fusion = simple concat
        user_vector = np.concatenate([text_vec, audio_vec, visual_vec])
        user_embeddings[user_id] = user_vector
    return user_embeddings

def save_embeddings(embeddings, filename):
    npy_path = os.path.join(OUTPUT_DIR, f"{filename}.npy")
    csv_path = os.path.join(OUTPUT_DIR, f"{filename}.csv")
    np.save(npy_path, embeddings, allow_pickle=True)
    df_out = pd.DataFrame([
        {"user_id": uid, **{f"dim_{i}": val for i, val in enumerate(vec)}}
        for uid, vec in embeddings.items()
    ])
    df_out.to_csv(csv_path, index=False)
    print(f"[INFO] Saved embeddings to:\n - {npy_path}\n - {csv_path}")

def main():
    # Load data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    print("[INFO] Loading journal data...")
    df = pd.read_csv(DATA_PATH)
    if "entry_text" not in df.columns or "user_id" not in df.columns:
        raise ValueError("CSV must have 'entry_text' and 'user_id' columns.")
    # Compute multimodal user embeddings
    user_embeddings = compute_user_embeddings(df, text_col="entry_text")
    save_embeddings(user_embeddings, "user_multimodal_embeddings")

if __name__ == "__main__":
    main()
