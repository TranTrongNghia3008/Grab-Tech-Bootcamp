from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
import csv
import io
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import pandas as pd

from app.schemas.dataset import DatasetFromConnection, DatasetId
from app.crud.connections import get_connection
from app.crud.datasets import create_dataset
from app.api.v1.dependencies import get_db
from app.utils.file_storage import save_dataframe_as_csv

# define router
router = APIRouter(
    prefix='/v1',
    tags=['datasets']
)

# Upload local CSV
@router.post('/datasets/')
async def upload_datasets(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Check if CSV or not
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='Only CSV files are supported.')

    # Read CSV file
    content = await file.read()
    decoded = content.decode('utf-8')
    df = pd.read_csv(io.StringIO(decoded))
    
    # Save to DB
    ds = create_dataset(db, connection_id=0, file_path="")
    filename = f"dataset_{ds.id}.csv"
    path = save_dataframe_as_csv(df, filename)
    
    # Upload record with path
    ds.file_path = path
    db.commit()
    db.refresh(ds)
    
    return DatasetId(id=ds.id)

# DB ingestion via query
@router.post('/datasets/from-connection/')
async def ingest_from_connection(payload: DatasetFromConnection, db: Session = Depends(get_db)):
    conn = get_connection(db, payload.connection_id)
    
    # Verify connection
    if not conn:
        raise HTTPException(status_code=404, detail='Connection not found')
    
    # Buil DB URL
    db_url =  (
        f'{conn.type}://{conn.username}:{conn.password}@'
        f'{conn.host}:{conn.port}/{conn.database}'
    )
    
    # Execute Query
    try:
        engine = create_engine(db_url)
        df = pd.read_sql(payload.query, con=engine)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Failed to execute query: {e}')
    
    # Create DB record and save CSV
    ds = create_dataset(db, connection_id=payload.connection_id, file_path="")
    filename = f"dataset_{ds.id}.csv"
    path = save_dataframe_as_csv(df, filename)
    
    # Upload record with path
    ds.file_path = path
    db.commit()
    db.refresh(ds)

    return DatasetId(id=ds.id)
    