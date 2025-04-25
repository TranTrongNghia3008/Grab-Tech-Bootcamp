from sqlalchemy import Column, Integer, String, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy import DateTime
from app.db.base import Base

class ModelJob(Base):
    __tablename__ = 'model_jobs'
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    model_type = Column(String, nullable=False)
    params = Column(JSON, nullable=True)
    status = Column(String, default="pending") # 'pending', 'running', 'completed', 'failed' 
    artifact_path = Column(String, nullable=True)
    prediction_path = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    datasets = relationship("Dataset", back_populates="model_jobs")