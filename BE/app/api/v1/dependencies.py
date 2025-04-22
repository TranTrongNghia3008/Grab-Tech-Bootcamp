from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings

# settings.database_url should be defined in `app/core/config.py`
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine
)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()