import os
from pathlib import Path
import pandas as pd

# Base directory for storing CSVs/Datasets
STORAGE_DIR = Path(os.getenv('DATA_STORAGE_DIR', 'data'))

def save_dataframe_as_csv(df: pd.DataFrame, filename: str) -> str:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = STORAGE_DIR / filename
    df.to_csv(path, index=False)
    return str(path)