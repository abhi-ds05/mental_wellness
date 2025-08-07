import os
import glob
import pandas as pd

def load_goemotions(goemotions_dir):
    """
    Loads and merges all GoEmotions CSV files in the specified directory.
    """
    csv_files = glob.glob(os.path.join(goemotions_dir, 'goemotions_*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No GoEmotions CSV files found in {goemotions_dir}")
    
    dataframes = []
    for f in csv_files:
        df = pd.read_csv(f)
        dataframes.append(df)
        print(f"Loaded {os.path.basename(f)} with shape {df.shape}")
    
    full_df = pd.concat(dataframes, ignore_index=True)
    print(f"Combined DataFrame shape: {full_df.shape}")
    return full_df

def main():
    # Robust path resolution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    goemotions_dir = os.path.join(current_dir, "..", "datasets", "GOEMOTIONS")
    goemotions_dir = os.path.abspath(goemotions_dir)

    goemotions_df = load_goemotions(goemotions_dir)

    print(goemotions_df.head())

    out_path = os.path.join(goemotions_dir, "goemotions_full.csv")
    goemotions_df.to_csv(out_path, index=False)
    print(f"Saved combined dataset to {out_path}")

if __name__ == "__main__":
    main()
