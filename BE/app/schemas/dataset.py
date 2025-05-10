from pydantic import BaseModel, Field
from typing import Any, List, Optional
from .commons import DataFrameStructure
from datetime import datetime

class DatasetFromConnection(BaseModel):
    connection_id: int
    query: str
    
class DatasetId(BaseModel):
    id: int
    
    class Config:
        from_attributes = True
        
class DatasetPreview(BaseModel):
    preview_data: DataFrameStructure = Field(..., description="Preview of the first 100 rows (or fewer if less data).")
    preview_row: int
    total_row: int
    total_col: int
    project_name: str
    
class DatasetInfo(BaseModel):
    """
    Information about a single dataset within a project.
    """
    dataset_id: int
    created_at: datetime

class DatasetFlatInfo(BaseModel):
    """
    Represents a single dataset with its project name, ID, and creation date.
    """
    id: int  # Corresponds to Dataset.id in your SQLAlchemy model
    project_name: Optional[str] # Corresponds to Dataset.project_name
    created_at: datetime # Corresponds to Dataset.created_at
    is_model: Optional[bool]
    is_clean: Optional[bool]

    class Config:
        from_attributes = True # Allows easy mapping from SQLAlchemy model instances

class DatasetFlatListResponse(BaseModel):
    """
    Response model for a flat list of datasets.
    """
    datasets: List[DatasetFlatInfo]
        
class FeatureProfile(BaseModel):
    """
    Profile information for a single feature (column) in the dataset.
    """
    feature_name: str
    dtype: str  # e.g., "integer", "float", "datetime", "categorical/string", "boolean"
    missing_percentage: float = Field(..., ge=0, le=100)
    unique_percentage: float = Field(..., ge=0, le=100)
    # Optional: could add min, max, mean for numerical, or top N categories for categorical
    # For now, sticking to the request.

class DatasetAnalysisReport(BaseModel):
    """
    Comprehensive analysis report for a dataset.
    """
    dataset_id: int
    project_name: Optional[str] = None
    file_path: str
    total_records: int
    total_features: int
    overall_missing_percentage: float = Field(..., ge=0, le=100)
    data_quality_score: Any # Can be a simple float or a more complex object/dict
    features: List[FeatureProfile]

    class Config:
        from_attributes = True # If you ever build this from an ORM object directly