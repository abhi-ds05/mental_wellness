import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_audio(file_path, sr=22050, n_mfcc=13):
    """
    Loads an audio file and extracts MFCC features.

    Args:
        file_path (str): Path to the audio file.
        sr (int): Target sampling rate.
        n_mfcc (int): Number of MFCCs to extract.

    Returns:
        np.ndarray: MFCC features.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        audio, _ = librosa.load(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return mfcc
    except Exception as e:
        raise RuntimeError(f"Failed to process audio file '{file_path}': {e}")

def visualize_mfcc(mfcc, title='MFCC'):
    """
    Displays a heatmap of the MFCC features.

    Args:
        mfcc (np.ndarray): MFCC features to visualize.
        title (str): Plot title.
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # === UPDATE THIS PATH TO YOUR ACTUAL .wav FILE ===
    file = 'E:/mental_wellness_ai/data/sample_audio.wav'

    try:
        mfcc_features = load_and_preprocess_audio(file)
        print(f"MFCC shape: {mfcc_features.shape}")
        visualize_mfcc(mfcc_features, title='MFCC Features')
    except Exception as err:
        print(f"[ERROR] {err}")
