from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.base import Base

class FinalizedModel(Base):
    __tablename__ = "finalized_models"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("automl_sessions.id", ondelete="CASCADE"), nullable=False)
    automl_job_id = Column(Integer, ForeignKey("automl_jobs.id"), nullable=True) # Job that created it
    model_name = Column(String, nullable=False)
    saved_model_path = Column(String, nullable=False)
    saved_metadata_path = Column(String, nullable=False)
    model_uri_for_registry = Column(String, nullable=True) # URI for MLflow etc.
    mlflow_run_id = Column(String, nullable=True) # Run ID from finalize step
    mlflow_registered_version = Column(Integer, nullable=True) # Version if registered
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationship back to session
    automl_sessions = relationship("AutoMLSession", back_populates="finalized_models")