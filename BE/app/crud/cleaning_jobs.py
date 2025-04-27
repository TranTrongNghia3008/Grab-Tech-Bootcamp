from sqlalchemy.orm import Session
from app.db.models.cleaning_jobs import CleaningJob

def get_cleaning_job_by_id(db: Session, job_id: int) -> CleaningJob | None:
    return db.query(CleaningJob).get(job_id)

def create_cleaning_job(db: Session, dataset_id: int, config: dict) -> CleaningJob:
    job = CleaningJob(dataset_id=dataset_id, config=config)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job

def update_cleaning_job(db: Session, job: CleaningJob, **kwargs) -> CleaningJob:
    for k, v in kwargs.items():
        setattr(job, k, v)
    
    db.commit()
    db.refresh(job)
    return job

def delete_cleaning_job(db: Session, job_id: int) -> bool:
    job = get_cleaning_job_by_id(db, job_id)
    if not job:
        return False

    db.delete(job)
    db.commit()
    return True