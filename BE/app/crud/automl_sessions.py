from typing import Dict, Any, Optional
from sqlalchemy.orm import Session # Sync Session

from app.crud.base import CRUDBase
from app.db.models.automl_sessions import AutoMLSession
from app.schemas.automl_sessions import AutoMLSessionCreate

AutoMLSessionUpdateSchema = Dict[str, Any]

class CRUDAutoMLSession(CRUDBase[AutoMLSession, AutoMLSessionCreate, AutoMLSessionUpdateSchema]):
    def update_status(
         self, db: Session, *, session_id: int, status: str, **kwargs: Any
     ) -> AutoMLSession | None:
         values_to_update = {"status": status, **kwargs}
         return self.update_atomic(db=db, id=session_id, values=values_to_update)

crud_automl_session = CRUDAutoMLSession(AutoMLSession)