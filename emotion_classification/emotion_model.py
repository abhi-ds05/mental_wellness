import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup

# Add data_processing to path so you can import clean_text.py
data_processing_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
if data_processing_path not in sys.path:
    sys.path.insert(0, data_processing_path)

try:
    from clean_text import clean_text
except ImportError:
    def clean_text(text):
        return text

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

class GoEmotionsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels.values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # remove batch dim
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float)
        }

class DistilBertForMultiLabelClassification(nn.Module):
    def __init__(self, n_classes):
        super(DistilBertForMultiLabelClassification, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.distilbert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        hidden_state = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)[0]  # (batch_size, seq_len, hidden_size)
        pooled_output = hidden_state[:, 0]  # CLS token representation is the first token
        output = self.drop(pooled_output)
        return self.out(output)

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler=None):
    model.train()
    losses = []
    true_labels = []
    pred_labels = []

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        preds = torch.sigmoid(outputs).cpu().detach().numpy()
        true_labels.append(labels.cpu().detach().numpy())
        pred_labels.append(preds)

    true_labels = np.vstack(true_labels)
    pred_labels = np.vstack(pred_labels)
    pred_labels_binary = (pred_labels >= 0.5).astype(int)
    f1 = f1_score(true_labels, pred_labels_binary, average='micro', zero_division=0)
    return np.mean(losses), f1

def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())

            preds = torch.sigmoid(outputs).cpu().detach().numpy()
            true_labels.append(labels.cpu().detach().numpy())
            pred_labels.append(preds)

    true_labels = np.vstack(true_labels)
    pred_labels = np.vstack(pred_labels)
    pred_labels_binary = (pred_labels >= 0.5).astype(int)
    f1 = f1_score(true_labels, pred_labels_binary, average='micro', zero_division=0)
    return np.mean(losses), f1, true_labels, pred_labels_binary

def get_dataloader(texts, labels, tokenizer, batch_size, max_len, shuffle=False):
    ds = GoEmotionsDataset(texts, labels, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def prepare_labels(df):
    exclude_cols = ['text', 'comment', 'id', 'subreddit', 'created_utc', 'parent_id']
    emotion_cols = [
        col for col in df.columns
        if col not in exclude_cols and df[col].nunique() <= 2 and pd.api.types.is_numeric_dtype(df[col])
    ]
    y = df[emotion_cols].fillna(0).astype(int)

    empty_cols = [col for col in emotion_cols if y[col].sum() == 0]
    if empty_cols:
        print(f"Dropping empty emotion columns: {empty_cols}")
        y = y.drop(columns=empty_cols)
        emotion_cols = [col for col in emotion_cols if col not in empty_cols]

    return y, emotion_cols

def main():
    # Configuration
    DATA_PATH = os.path.join('datasets', 'GOEMOTIONS', 'goemotions_full.csv')
    MODEL_SAVE_DIR = os.path.join('emotion_classification', 'models')
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 1  # train only 1 epoch initially
    SUBSAMPLE_SIZE = 50000  # Set to None or 20000 or 50000 for prototyping

    # Load data
    df = pd.read_csv(DATA_PATH)

    # Subsample dataset if requested
    if SUBSAMPLE_SIZE is not None and SUBSAMPLE_SIZE < len(df):
        print(f"Subsampling dataset to {SUBSAMPLE_SIZE} rows for prototyping.")
        df = df.sample(n=SUBSAMPLE_SIZE, random_state=42).reset_index(drop=True)

    text_column = 'text' if 'text' in df.columns else 'comment'
    print(f"Loaded data shape: {df.shape}")

    # Clean text
    df[text_column] = df[text_column].apply(lambda x: clean_text(str(x)))

    # Prepare labels
    y, emotion_labels = prepare_labels(df)
    print(f"Using emotion labels: {emotion_labels}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(df[text_column], y, test_size=0.2, random_state=42)
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_loader = get_dataloader(X_train, y_train, tokenizer, BATCH_SIZE, MAX_LEN, shuffle=True)
    val_loader = get_dataloader(X_val, y_val, tokenizer, BATCH_SIZE, MAX_LEN)

    model = DistilBertForMultiLabelClassification(len(emotion_labels))
    model.to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_f1 = 0
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss, train_f1 = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE, scheduler)
        print(f"Train loss: {train_loss:.4f}, Train micro-F1: {train_f1:.4f}")

        val_loss, val_f1, y_true, y_pred = eval_model(model, val_loader, loss_fn, DEVICE)
        print(f"Val loss: {val_loss:.4f}, Val micro-F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'emotion_labels': emotion_labels,
                'tokenizer': tokenizer
            }, os.path.join(MODEL_SAVE_DIR, 'distilbert_emotion_model.bin'))
            print(f"Saved best model at epoch {epoch+1}")

    print("\nFinal classification report on validation set:")
    print(classification_report(y_true, y_pred, target_names=emotion_labels, zero_division=0))

if __name__ == "__main__":
    main()
