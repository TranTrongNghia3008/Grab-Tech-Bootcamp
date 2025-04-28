from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.base import Base

class AutoMLJob(Base):
    __tablename__ = "automl_jobs"

    id = Column(Integer, primary_key=True, index=True)
    # Can be null for jobs not tied to a session (predict, explain)
    session_id = Column(Integer, ForeignKey("automl_sessions.id", ondelete="CASCADE"), nullable=True)
    job_type = Column(String, index=True, nullable=False)
    status = Column(String, index=True, default="pending", nullable=False)
    input_params = Column(JSON, nullable=True)
    results = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationship
    automl_sessions = relationship("AutoMLSession", back_populates="automl_jobs")