import os
import pandas as pd

def load_journal_entries(journal_path):
    """
    Loads the journal entries CSV file.

    Args:
        journal_path (str): Path to the journal_entries.csv file.
    Returns:
        pd.DataFrame: Loaded DataFrame containing journal entries.
    Raises:
        FileNotFoundError: If the journal CSV is not found.
        pd.errors.EmptyDataError: If the CSV is empty.
    """
    abs_path = os.path.abspath(journal_path)
    print(f"Resolved path: {abs_path}")

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"No journal entries found at {abs_path}")
    
    df = pd.read_csv(abs_path)
    
    if df.empty:
        raise pd.errors.EmptyDataError(f"{abs_path} is empty!")
    
    print(f"Loaded {os.path.basename(journal_path)} with shape {df.shape}")
    print("Columns:", list(df.columns))
    return df

def validate_journal_df(df, required_columns=None):
    """
    Checks for missing required columns and basic data issues.
    Args:
        df (pd.DataFrame): The loaded journal DataFrame.
        required_columns (list or None): List of required column names.
    Returns:
        pd.DataFrame: Cleaned (if needed) DataFrame.
    """
    if required_columns is None:
        required_columns = ["user_id", "timestamp", "entry_text", "mood_score", "emotion"]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in journal data: {missing_cols}")

    # Basic cleaning
    df = df.drop_duplicates()
    df = df.fillna({'mood_score': 0, 'emotion': 'Unknown'})
    
    print(f"Validated journal data: {df.shape[0]} entries, {len(df.columns)} columns.")
    return df

def main():
    # Dynamically construct the dataset path regardless of where script is run
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    journal_path = os.path.join(project_root, 'datasets', 'synthetic_user_journals', 'journal_entries.csv')

    # 1. Load the data
    journals_df = load_journal_entries(journal_path)

    # 2. Validate essential columns/structure
    journals_df = validate_journal_df(journals_df)

    # 3. Print sample rows
    print(journals_df.head())

    # 4. Optionally save a cleaned/validated version
    cleaned_path = os.path.join(os.path.dirname(journal_path), "journal_entries_cleaned.csv")
    journals_df.to_csv(cleaned_path, index=False)
    print(f"✅ Saved cleaned journal data to {cleaned_path}")

if __name__ == "__main__":
    main()
