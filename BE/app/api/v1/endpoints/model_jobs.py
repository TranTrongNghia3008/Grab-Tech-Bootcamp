from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.api.v1.dependencies import get_db
from app.db.models.model_jobs import ModelJob
from app.crud.model_jobs import get_job_by_id

router = APIRouter(
    prefix='/v1',
    tags=['models']
)

@router.get('/models/{job_id}/artifacts')
def download_artifacts(job_id: int, db: Session = Depends(get_db)):
    job = get_job_by_id(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return FileResponse(path=job.artifact_path, filename=f'artifact_{job_id}.zip', media_type='application/zip')

@router.get('/models/{job_id}/predictions')
def download_predictions(job_id: int, db: Session = Depends(get_db)):
    job = get_job_by_id(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return FileResponse(path=job.prediction_path, filename=f'prediction_{job_id}.csv', media_type='text/csv')