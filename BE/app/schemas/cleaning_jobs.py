from pydantic import BaseModel
from typing import Dict, Any, Literal, Optional

class CleaningConfig(BaseModel):
    remove_duplicates: bool = False
    handle_missing_values: bool = False
    smooth_noisy_data: bool = False
    handle_outliers: bool = False
    reduce_cardinality: bool = False
    encode_categorical_values: bool = False
    feature_scaling: bool = False

class CleaningPreview(BaseModel):
    missing: Dict[str, int]
    outliers: Dict[str, int]
    duplicates: int

class CleaningStatus(BaseModel):
    status: Literal['pending','running','completed','failed']

class CleaningResults(BaseModel):
    original_rows: int
    cleaned_rows: int
    cleaned_dataset_id: int
    cleaned_file_path: str
    error: Optional[str] = None 

class CleaningJobOut(BaseModel):
    id: int
    dataset_id: int
    status: str
    config: Optional[Dict[str, Any]]
    results: Optional[Dict[str, Any]]

    class Config:
        from_attributes = True