from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.api.v1.dependencies import get_db
from app.schemas.cleaning_jobs import (
    CleaningConfig, CleaningPreview,
    CleaningStatus, CleaningResults, CleaningJobOut, CleaningDataPreview
)
from app.services.cleaning_service import (
    schedule_cleaning, run_cleaning_job,
    preview_cleaning, get_status, get_results,
    update_cleaning, delete_cleaning
)

from app.crud.cleaning_jobs import crud_cleaning_job
from app.utils.file_storage import load_csv_as_dataframe, get_file_path
from app import schemas
import pandas as pd

router = APIRouter(prefix="/v1", tags=["cleaning"])

@router.get("/datasets/{dataset_id}/cleaning/preview", response_model=CleaningPreview)
def preview_issues(dataset_id: int, db: Session = Depends(get_db)):
    return preview_cleaning(dataset_id, db)

@router.post("/datasets/{dataset_id}/cleaning", response_model=CleaningJobOut)
def post_cleaning(
    dataset_id: int,
    config: CleaningConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    job_id = schedule_cleaning(dataset_id, config.dict(), db)
    background_tasks.add_task(run_cleaning_job, job_id)
    job = crud_cleaning_job.get(db, job_id)
    return job

@router.get("/cleaning/{job_id}/status", response_model=CleaningStatus)
def cleaning_status(job_id: int, db: Session = Depends(get_db)):
    status = get_status(job_id, db)
    if status is None:
        raise HTTPException(404, "Job not found")
    return {"status": status}

@router.get("/cleaning/{job_id}/results", response_model=CleaningResults)
def cleaning_results(job_id: int, db: Session = Depends(get_db)):
    results = get_results(job_id, db)
    if results is None:
        raise HTTPException(404, "Results not found")
    return {"job_id": job_id, **results}

@router.get("/datasets/{dataset_id}/cleaned_data", response_model=CleaningDataPreview)
def get_cleaned_data(dataset_id: int, db: Session = Depends(get_db)):
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
    
    response_data = CleaningDataPreview(
        preview_cleaned=preview_schema,
        preview_row=preview_row,
        total_row=total_row
    )
    
    return response_data
    

@router.put("/cleaning/{job_id}")
def put_cleaning(job_id: int, config: CleaningConfig, db: Session = Depends(get_db)):
    if not update_cleaning(job_id, config.dict(), db):
        raise HTTPException(404, "Job not found")
    return {"detail": "Config updated"}

@router.delete("/cleaning/{job_id}")
def delete_clean(job_id: int, db: Session = Depends(get_db)):
    if not delete_cleaning(job_id, db):
        raise HTTPException(404, "Job not found")
    return {"detail": "Job deleted"}