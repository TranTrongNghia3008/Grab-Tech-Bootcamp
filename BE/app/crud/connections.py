from sqlalchemy.orm import Session
from app.db.models.connections import Connection

def get_connection(db: Session, conn_id: int) -> Connection | None:
    return db.query(Connection).filter(Connection.id == conn_id).first()