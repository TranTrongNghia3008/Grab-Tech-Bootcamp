from sqlalchemy.orm import Session 
from app.db.models.datasets import Dataset
from typing import Optional
from app.crud.base import CRUDBase
from pydantic import BaseModel

def create_dataset(db: Session, connection_id: int, file_path: str) -> Dataset:
    ds = Dataset(connection_id=connection_id, file_path=file_path)
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds

def get_dataset_by_id(db: Session, dataset_id: int) -> Dataset | None:
    return db.query(Dataset).get(dataset_id)


class DatasetCreateSchema(BaseModel):
    file_path: str # Example required field
    connection_id: Optional[int] = None

class DatasetUpdateSchema(BaseModel):
    file_path: Optional[str] = None
    connection_id: Optional[int] = None


# --- Using CRUDBase ---
class CRUDDataset(CRUDBase[Dataset, DatasetCreateSchema, DatasetUpdateSchema]):
    def get(self, db: Session, id: int) -> Optional[Dataset]:
        """
        Retrieves a Dataset record by its primary key (ID).
        """
        # The base class get method handles this query:
        # return db.query(self.model).filter(self.model.id == id).first()
        return super().get(db, id)
    
crud_dataset = CRUDDataset(Dataset)