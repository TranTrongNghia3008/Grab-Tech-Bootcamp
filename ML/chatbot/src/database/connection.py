from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import Engine
from src.config import DB_URI
import sys

def get_db_engine() -> Engine:
    """Creates and returns a SQLAlchemy engine instance."""
    try:
        engine = create_engine(DB_URI, pool_pre_ping=True)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Database connection successful.")
        return engine
    except Exception as e:
        print(f"Database connection failed: {e}", file=sys.stderr)
        sys.exit(1) # Exit if DB connection fails on startup

# Create a single engine instance to be reused
engine = get_db_engine()

# Optional: If you plan to use sessions later
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)