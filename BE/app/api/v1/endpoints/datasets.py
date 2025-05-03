from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
import io
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import pandas as pd
from app.crud.datasets import DatasetCreateSchema, DatasetUpdateSchema
from app.schemas.dataset import DatasetFromConnection, DatasetId, DatasetPreview

from app.crud.datasets import crud_dataset
from app.crud.connections import get_connection_by_id

from app.api.v1.dependencies import get_db
from app.utils.file_storage import save_dataframe_as_csv, get_file_path, load_csv_as_dataframe
from app import schemas

router = APIRouter(
    prefix='/v1',
    tags=['datasets']
)

@router.post('/datasets/', response_model=DatasetId)
async def upload_datasets(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='Only CSV files are supported.')

    try:
        content = await file.read()
        decoded = content.decode('utf-8')
        df = pd.read_csv(io.StringIO(decoded))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")

    # 1. Create initial DB record
    dataset_in_create = DatasetCreateSchema(connection_id=None)
    ds = crud_dataset.create(db=db, obj_in=dataset_in_create)

    # 2. Save the dataframe using the dataset ID
    filename = f"dataset_{ds.id}.csv"
    try:
        path = save_dataframe_as_csv(df, filename)
    except Exception as e:
        # Cleanup the created DB record if file saving fails
        crud_dataset.remove(db=db, id=ds.id)
        raise HTTPException(status_code=500, detail=f"Failed to save dataset file: {e}")

    # 3. Update the DB record with the file path
    dataset_in_update = DatasetUpdateSchema(file_path=path)
    ds_updated = crud_dataset.update(db=db, db_obj=ds, obj_in=dataset_in_update)

    return DatasetId(id=ds_updated.id)

@router.post('/datasets/from-connection/', response_model=DatasetId)
async def ingest_from_connection(payload: DatasetFromConnection, db: Session = Depends(get_db)):
    conn = get_connection_by_id(db, payload.connection_id)
    if not conn:
        raise HTTPException(status_code=404, detail='Connection not found')

    try:
        db_url = (
            f'{conn.type}://{conn.username}:{conn.password}@'
            f'{conn.host}:{conn.port}/{conn.database}'
        )
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error building database URL: {e}")

    engine = None # Ensure engine is defined for finally block
    try:
        engine = create_engine(db_url)
        with engine.connect() as connection:
             df = pd.read_sql(payload.query, con=connection)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Failed to execute query or read data: {e}')
    finally:
        if engine:
            engine.dispose() # Release connection pool resources

    # 1. Create initial DB record
    dataset_in_create = DatasetCreateSchema(connection_id=payload.connection_id)
    ds = crud_dataset.create(db=db, obj_in=dataset_in_create)

    # 2. Save the dataframe
    filename = f"dataset_{ds.id}.csv"
    try:
        path = save_dataframe_as_csv(df, filename)
    except Exception as e:
        # Cleanup DB record if saving fails
        crud_dataset.remove(db=db, id=ds.id)
        raise HTTPException(status_code=500, detail=f"Failed to save dataset file: {e}")

    # 3. Update the DB record with the path
    dataset_in_update = DatasetUpdateSchema(file_path=path)
    ds_updated = crud_dataset.update(db=db, db_obj=ds, obj_in=dataset_in_update)

    return DatasetId(id=ds_updated.id)

@router.get('/datasets/{dataset_id}/preview', response_model=DatasetPreview)
def get_dataset_preview(dataset_id: int):
    file_path = get_file_path(f'dataset_{dataset_id}_cleaned.csv')
    result_df = load_csv_as_dataframe(file_path)
    preview_row = min(100, len(result_df))
    preview_df = result_df.head(preview_row).copy()
    total_row = len(result_df)
    
    try:
        # Ensure index is reset if needed, handle potential NaNs for JSON conversion
        df_for_schema = preview_df.reset_index(drop=True)
        data_list = df_for_schema.astype(object).where(pd.notnull(df_for_schema), None).values.tolist()
        preview_schema = schemas.DataFrameStructure(
            columns=df_for_schema.columns.tolist(),
            data=data_list
        )
    except Exception as convert_e:
        # Log the error internally if needed
        print(f"Error converting preview DataFrame to schema: {convert_e}")
        raise HTTPException(status_code=500, detail="Failed to format prediction preview results.")
    
    response_data = DatasetPreview(
        preview_data=preview_schema,
        preview_row=preview_row,
        total_row=total_row
    )
    
    return response_data