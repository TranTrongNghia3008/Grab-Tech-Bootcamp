from sqlalchemy.orm import Session
from typing import Optional, List
from pydantic import BaseModel

from app.db.models.datasets import Dataset
from app.crud.base import CRUDBase

# --- Schemas ---
# Define data structures for creating and updating Dataset records via the API/CRUD operations.
class DatasetCreateSchema(BaseModel):
    file_path: str = ""
    connection_id: Optional[int] = None

class DatasetUpdateSchema(BaseModel):
    file_path: Optional[str] = None
    connection_id: Optional[int] = None

# --- CRUD Class using CRUDBase ---
class CRUDDataset(CRUDBase[Dataset, DatasetCreateSchema, DatasetUpdateSchema]):
    pass

# --- Instantiate the CRUD class ---
crud_dataset = CRUDDataset(Dataset)