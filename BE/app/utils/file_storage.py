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