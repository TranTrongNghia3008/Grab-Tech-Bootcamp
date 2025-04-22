from pydantic import BaseModel
from typing import Any

class DatasetFromConnection(BaseModel):
    connection_id: int
    query: str
    
class DatasetId(BaseModel):
    id: int
    
    class Config:
        orm_mode = True