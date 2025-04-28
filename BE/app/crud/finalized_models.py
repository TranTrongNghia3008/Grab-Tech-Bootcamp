from typing import Dict, Any, Optional
from sqlalchemy.orm import Session # Sync Session
from pydantic import BaseModel

from app.crud.base import CRUDBase
from app.db.models.finalized_models import FinalizedModel

class FinalizedModelCreateInternal(BaseModel):
    session_id: int; 
    automl_job_id: Optional[int]=None; 
    model_name: str; saved_model_path: str; 
    saved_metadata_path: str; 
    model_uri_for_registry: Optional[str]=None; 
    mlflow_run_id: Optional[str]=None; 
    mlflow_registered_version: Optional[int]=None
FinalizedModelUpdateSchema = Dict[str, Any]

class CRUDFinalizedModel(CRUDBase[FinalizedModel, FinalizedModelCreateInternal, FinalizedModelUpdateSchema]):
    def update_registration(
         self, db: Session, *, model_id: int, mlflow_version: int, model_uri: Optional[str] = None
    ) -> FinalizedModel | None:
        values = {"mlflow_registered_version": mlflow_version}
        if model_uri: 
            values["model_uri_for_registry"] = model_uri
        return self.update_atomic(db=db, id=model_id, values=values)

crud_finalized_model = CRUDFinalizedModel(FinalizedModel)