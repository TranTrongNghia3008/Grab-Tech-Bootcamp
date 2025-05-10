from sqlalchemy.orm import Session, joinedload
from sqlalchemy.sql import func
from typing import Optional, List, Dict, Any
import json # For loading/dumping if needed, though SQLAlchemy handles JSON type

from app.db.models.chatbot_sessions import ChatSessionState, JourneyLogEntry
import logging
# from app.db.models.datasets import Dataset # If needed for direct reference

SESSION_TIMEOUT_SECONDS_DB: int = 3600 * 24 # Example: 24 hours for DB persistence

logger = logging.getLogger(__name__)

def get_active_session_state(db: Session, dataset_id: int, session_uuid: str) -> Optional[ChatSessionState]:
    session = db.query(ChatSessionState)\
        .options(joinedload(ChatSessionState.journey_log_entries))\
        .filter(ChatSessionState.dataset_id == dataset_id, ChatSessionState.session_uuid == session_uuid)\
        .first()
    if session:
        session.last_accessed_at = func.now()
        db.commit()
        # db.refresh(session) # Only if you need to immediately see the updated last_accessed_at
    return session

def create_session_state(
    db: Session,
    dataset_id: int,
    session_uuid: str,
    initial_chat_history: Optional[List[Dict[str, Any]]] = None,
    auto_execute_enabled: bool = False
) -> ChatSessionState:
    db_session_state = ChatSessionState(
        dataset_id=dataset_id,
        session_uuid=session_uuid,
        chat_history_json=initial_chat_history if initial_chat_history else [],
        analysis_journey_log_json=[],
        auto_execute_enabled=auto_execute_enabled
        # Other fields will use defaults or be None
    )
    db.add(db_session_state)
    db.commit()
    db.refresh(db_session_state)
    return db_session_state

def update_session_state_from_service(
    db: Session,
    db_session_state: ChatSessionState, # The ORM object
    chat_history: List[Dict[str, Any]],
    analysis_journey_log: List[Dict[str, Any]],
    current_focus_filter: Optional[str],
    pending_code_json: Optional[Dict[str, Any]],
    pending_whatif_json: Optional[Dict[str, Any]],
    pending_focus_proposal_json: Optional[Dict[str, Any]],
    last_plot_path: Optional[str],
    auto_execute_enabled: bool # Persist this preference
) -> ChatSessionState:
    db_session_state.chat_history_json = chat_history
    db_session_state.analysis_journey_log_json = analysis_journey_log
    db_session_state.current_focus_filter = current_focus_filter
    db_session_state.pending_code_to_execute_json = pending_code_json
    db_session_state.pending_whatif_code_to_execute_json = pending_whatif_json
    db_session_state.pending_focus_proposal_json = pending_focus_proposal_json
    db_session_state.last_executed_plot_path = last_plot_path
    db_session_state.auto_execute_enabled = auto_execute_enabled
    # last_accessed_at is updated automatically by onupdate=func.now()
    
    db.commit()
    #db.refresh(db_session_state)
    return db_session_state

def add_journey_log_db_entry(
    db: Session,
    chat_session_state_orm: ChatSessionState, # Pass the parent ORM session state
    event_type: str,
    payload: Optional[Dict[str, Any]] = None
) -> JourneyLogEntry:
    db_log_entry = JourneyLogEntry(
        chat_session_state_id=chat_session_state_orm.id,
        event_type=event_type.upper(), # Good to standardize case
        payload_json=payload
    )
    db.add(db_log_entry)
    # The relationship ChatSessionState.journey_log_entries will automatically update
    # when the session is reloaded or if you append to it and commit.
    # For an immediate commit of the log entry:
    db.commit()
    db.refresh(db_log_entry) # To get its ID and timestamp from DB
    logger.info(f"Logged journey for session {chat_session_state_orm.session_uuid}: {event_type}")
    return db_log_entry

def delete_expired_session_states(db: Session) -> int:
    """
    Deletes session states where last_accessed_at is older than SESSION_TIMEOUT_SECONDS_DB.
    Returns the number of deleted sessions.
    NOTE: Date/time arithmetic is database-specific. This is a generic placeholder.
    For PostgreSQL: WHERE last_accessed_at < NOW() - INTERVAL 'X seconds'
    For SQLite: WHERE julianday('now') - julianday(last_accessed_at) * 86400.0 > X
    """
    # This is a simplified delete for example purposes.
    # A robust solution would use proper interval arithmetic for your DB.
    # For now, this won't actually delete based on time, just shows the structure.
    # You'd need to implement the correct WHERE clause.
    # Example for PostgreSQL (conceptual, assuming SESSION_TIMEOUT_SECONDS_DB is in seconds):
    # from sqlalchemy import text
    # result = db.execute(
    #     text("DELETE FROM chatbot_session_states WHERE last_accessed_at < NOW() - INTERVAL ':timeout seconds'").bindparams(timeout=SESSION_TIMEOUT_SECONDS_DB)
    # )
    # num_deleted = result.rowcount
    
    # For now, let's just log that this would be called.
    # Implement actual deletion based on your DB.
    num_deleted = 0 
    # query = db.query(ChatSessionState).filter(...) # Add your DB-specific time filter
    # num_deleted = query.delete(synchronize_session=False)
    # db.commit()
    logging.info(f"Placeholder for deleting expired sessions. Would delete {num_deleted} sessions.")
    return num_deleted

def get_sessions_for_dataset(db: Session, dataset_id: int) -> List[ChatSessionState]:
    """
    Retrieves all ChatSessionState records associated with a given dataset_id.
    Orders by last_accessed_at descending by default.
    """
    return db.query(ChatSessionState)\
        .filter(ChatSessionState.dataset_id == dataset_id)\
        .order_by(ChatSessionState.last_accessed_at.desc())\
        .all()