from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.api.v1.dependencies import get_db
from app.schemas.cleaning_jobs import (
    CleaningConfig, CleaningPreview,
    CleaningStatus, CleaningResults, CleaningJobOut
)
from app.services.cleaning_service import (
    schedule_cleaning, run_cleaning_job,
    preview_cleaning, get_status, get_results,
    update_cleaning, delete_cleaning
)

from app.crud.cleaning_jobs import get_cleaning_job_by_id

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
    job = get_cleaning_job_by_id(db, job_id)
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