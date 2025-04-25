from sqlalchemy.orm import Session
from app.db.models.model_jobs import ModelJob

def get_job_by_id(db: Session, job_id: int) -> ModelJob | None:
    return db.query(ModelJob).filter(ModelJob.id == job_id).first()