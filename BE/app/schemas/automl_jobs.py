from pydantic import BaseModel, UUID4
from typing import Optional, Dict, Any
import datetime
from app.schemas.commons import StatusResponse # Use common response

class AutoMLJobResponse(BaseModel):
    id: int
    session_id: Optional[UUID4]
    job_type: str
    status: str
    error_message: Optional[str]
    created_at: datetime.datetime
    started_at: Optional[datetime.datetime]
    completed_at: Optional[datetime.datetime]
    input_params: Optional[Dict[str, Any]]
    class Config: 
        from_attributes = True

class AutoMLJobResultBase(StatusResponse):
    input_params: Optional[Dict[str, Any]] = None
    pass