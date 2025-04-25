from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.v1.dependencies import get_db
from app.db.models.datasets import Dataset
from app.utils.file_storage import load_csv_as_dataframe

router = APIRouter(
    prefix='/v1/datasets', 
    tags=['EDA']
)

# Get summary statistic from dataset
@router.get('/{dataset_id}/eda/stats')
def get_summary_statistics(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail='Dataset not found')
    
    df = load_csv_as_dataframe(dataset.file_path)
    return df.describe(include='all').fillna('').to_dict()

# Get correlation matrices from dataset
@router.get('/{dataset_id}/eda/corr')
def get_correlation_matrix(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(status_code=404, details='Dataset not found')
    
    df = load_csv_as_dataframe(dataset.file_path)
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr().fillna(0)
    return corr.to_dict()