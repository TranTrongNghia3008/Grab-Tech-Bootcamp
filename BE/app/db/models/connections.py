from sqlalchemy import Column, Integer, String
from app.db.base import Base

class Connection(Base):
    __tablename__ = 'connections'
    
    id = Column(Integer, primary_key=True, index=True)
    type = Column(String, nullable=False) # postgres or ...
    host = Column(String, nullable=False)
    port = Column(Integer, nullable=False)
    database = Column(String, nullable=False)
    username = Column(String, nullable=False)
    password = Column(String, nullable=False)