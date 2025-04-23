from pydantic import BaseModel

class ConnectionId(BaseModel):
    id: int
    
    class Config:
        orm_mode = True

class ConnectionOut(BaseModel):
    id: int
    type: str
    host: str
    port: int
    database: str
    username: str
    
    class Config:
        orm_mode = True
        
class ConnectionCreate(BaseModel):
    type: str
    host: str
    port: int
    database: str
    username: str
    password: str