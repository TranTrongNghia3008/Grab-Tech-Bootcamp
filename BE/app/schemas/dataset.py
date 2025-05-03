from pydantic import BaseModel, Field
from typing import Any
from .commons import DataFrameStructure

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