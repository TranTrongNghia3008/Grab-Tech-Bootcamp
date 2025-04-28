# backend/app/schemas/finalize.py
from pydantic import BaseModel, Field, field_validator, UUID4
from typing import Optional, Dict, Any
import datetime
from app.schemas.automl_jobs import AutoMLJobResultBase

class FinalizeModelRequest(BaseModel):
    source_model_id: str = Field(...)
    model_name: str = Field(...)
    register_in_mlflow: bool = Field(False)
    mlflow_registered_model_name: Optional[str] = Field(None)

    @field_validator('mlflow_registered_model_name', mode='before', always=True)
    def check_mlflow_name(cls, v, values):
        if values.get('register_in_mlflow') and not v:
            raise ValueError('mlflow_registered_model_name is required')
        return v

class FinalizeResult(AutoMLJobResultBase):
    finalized_model_db_id: Optional[int] = None
    saved_model_path: Optional[str] = None
    saved_metadata_path: Optional[str] = None
    model_uri_for_registry: Optional[str] = None
    mlflow_registered_version: Optional[int] = None

class FinalizedModelResponse(BaseModel):
    id: int
    session_id: int
    model_name: str
    saved_model_path: str
    model_uri_for_registry: Optional[str] = None
    mlflow_registered_version: Optional[int] = None
    created_at: datetime.datetime

    class Config:
        orm_mode = True