from pydantic import BaseModel
from typing import Dict, Any, Literal, Optional

class CleaningConfig(BaseModel):
    impute_missing: Optional[Dict[str, Any]] = None
    filter_outliers: Optional[Dict[str, Any]] = None
    drop_duplicates: bool = True
    enforce_types: Optional[Dict[str, str]] = None

class CleaningPreview(BaseModel):
    missing: Dict[str, int]
    outliers: Dict[str, int]
    duplicates: int

class CleaningStatus(BaseModel):
    status: Literal['pending','running','completed','failed']

class CleaningResults(BaseModel):
    job_id: int
    cleaned_dataset_id: int
    original_rows: int
    cleaned_rows: int
    details: Dict[str, Any]

class CleaningJobOut(BaseModel):
    id: int
    dataset_id: int
    status: str
    config: Optional[Dict[str, Any]]
    results: Optional[Dict[str, Any]]

    class Config:
        orm_mode = True