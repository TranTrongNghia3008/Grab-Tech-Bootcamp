import datetime
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session # Sync Session
from pydantic import BaseModel

from app.crud.base import CRUDBase
from app.db.models.automl_jobs import AutoMLJob

class AutoMLJobCreateInternal(BaseModel):
    job_type: str; 
    session_id: Optional[int] = None; 
    input_params: Optional[Dict[str, Any]] = None
AutoMLJobUpdateSchema = Dict[str, Any]

class CRUDAutoMLJob(CRUDBase[AutoMLJob, AutoMLJobCreateInternal, AutoMLJobUpdateSchema]):
    def update_status(
        self, db: Session, *, job_id: int, status: str, **kwargs: Any
    ) -> AutoMLJob | None:
        values_to_update = {"status": status}
        now = datetime.datetime.now(datetime.timezone.utc)
        if status == "running" and "started_at" not in kwargs: 
            values_to_update["started_at"] = now
        if status in ["completed", "failed"] and "completed_at" not in kwargs:
            values_to_update["completed_at"] = now
        values_to_update.update(kwargs)
        return self.update_atomic(db=db, id=job_id, values=values_to_update)

    def update_results(
         self, db: Session, *, job_id: int, results: Dict[str, Any], status: str = "completed"
     ) -> AutoMLJob | None:
        return self.update_status(db, job_id=job_id, status=status, results=results)

    def update_error(
         self, db: Session, *, job_id: int, error_message: str
     ) -> AutoMLJob | None:
        return self.update_status(db, job_id=job_id, status="failed", error_message=error_message)

crud_automl_job = CRUDAutoMLJob(AutoMLJob)