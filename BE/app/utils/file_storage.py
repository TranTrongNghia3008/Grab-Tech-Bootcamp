import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Base directory for storing CSVs/Datasets
STORAGE_DIR = Path(os.getenv('DATA_STORAGE_DIR', 'data'))

# Save DFs to server as CSVs
def save_dataframe_as_csv(df: pd.DataFrame, filename: str) -> str:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = STORAGE_DIR / filename
    df.to_csv(path, index=False)
    return str(path)

def get_file_path(filename: str) -> str:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = STORAGE_DIR / filename
    return str(path)

# Load CSV from DS
def load_csv_as_dataframe(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def get_cleaned_df_path(raw_file_path: str) -> str:
    file_path = Path(raw_file_path)
    cleaned_name = file_path.stem + "_cleaned" + file_path.suffix
    cleaned_file_path = file_path.with_name(cleaned_name)
    
    file_path = Path(cleaned_file_path)
    
    if not file_path.exists():
        df = load_csv_as_dataframe(raw_file_path)
        save_dataframe_as_csv(df, cleaned_name)
    
    return cleaned_file_path