from pydantic import BaseModel, Field, UUID4
from typing import Optional, List, Dict, Any
import datetime

class AutoMLSessionBase(BaseModel):
    name: Optional[str] = None
    dataset_id: str
    target_column: str
    feature_columns: Optional[List[str]] = None
    mlflow_experiment_name: Optional[str] = None

class AutoMLSessionCreate(AutoMLSessionBase):
    train_size: float = Field(0.8, gt=0, lt=1)

class AutoMLSessionResponse(BaseModel):
    id: int
    name: Optional[str] = None
    dataset_id: str
    target_column: str
    task_type: Optional[str]
    status: str
    data_profile_report_url: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    mlflow_setup_run_id: Optional[str]
    created_at: datetime.datetime
    class Config: orm_mode = True

class AutoMLSessionErrorResponse(BaseModel): detail: str