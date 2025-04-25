from sqlalchemy.orm import Session 
from app.db.models.datasets import Dataset

def create_dataset(db: Session, connection_id: int, file_path: str) -> Dataset:
    ds = Dataset(connection_id=connection_id, file_path=file_path)
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds