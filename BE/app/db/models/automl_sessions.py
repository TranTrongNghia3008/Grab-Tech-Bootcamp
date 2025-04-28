from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.base import Base

class AutoMLSession(Base):
    __tablename__ = 'automl_sessions'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=True)
    dataset_id = Column(String, nullable=False) # Refers to your dataset concept
    target_column = Column(String, nullable=False)
    feature_columns = Column(JSON, nullable=True)
    task_type = Column(String, nullable=True)
    status = Column(String, index=True, default="initialized", nullable=False)
    data_profile_report_url = Column(String, nullable=True)
    mlflow_experiment_name = Column(String, nullable=True)
    mlflow_setup_run_id = Column(String, nullable=True)
    config = Column(JSON, nullable=True) # Store setup config (paths, train_size etc)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships (Standard Naming)
    automl_jobs = relationship("AutoMLJob", back_populates="automl_sessions", cascade="all, delete-orphan")
    finalized_models = relationship("FinalizedModel", back_populates="automl_session", cascade="all, delete-orphan")