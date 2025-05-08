from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy import DateTime
from sqlalchemy.orm import relationship
from app.db.base import Base

class Dataset(Base):
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    connection_id = Column(Integer, ForeignKey("connections.id"), nullable=True)
    file_path = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    connections = relationship("Connection", back_populates="datasets")
    cleaning_jobs = relationship('CleaningJob', back_populates='datasets', cascade='all, delete-orphan')
    automl_sessions = relationship('AutoMLSession', back_populates='datasets', cascade='all, delete-orphan')
