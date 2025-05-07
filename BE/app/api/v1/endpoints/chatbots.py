from fastapi import APIRouter, Depends, HTTPException, Header, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Tuple, Optional
import time
import uuid # For generating session IDs
import logging
import os
import random # For opportunistic background task scheduling
from app.api.v1.dependencies import get_db
from app.crud.datasets import crud_dataset
from app.schemas import chatbots as chatbot_schemas
from app.services.chatbot_service import ChatbotService # Your ChatbotService

router = APIRouter()
logger = logging.getLogger(__name__)

# --- In-Memory Session Management ---
# Key: Tuple(dataset_id: int, session_id: str)
# Value: Dict{"service": ChatbotService, "last_accessed": float (timestamp)}
active_chatbot_sessions: Dict[Tuple[int, str], Dict[str, any]] = {}
SESSION_TIMEOUT_SECONDS: int = 3600  # 1 hour

# --- Session Management Helper Functions ---
def _get_or_create_session_service(
    dataset_id: int,
    session_id: str,
    db: Session # Pass db session for fetching dataset info
) -> ChatbotService:
    """
    Internal helper to retrieve or create a ChatbotService instance for a session.
    Manages the active_chatbot_sessions dictionary.
    """
    session_key = (dataset_id, session_id)
    current_time = time.time()

    if session_key in active_chatbot_sessions:
        session_data = active_chatbot_sessions[session_key]
        if (current_time - session_data["last_accessed"]) < SESSION_TIMEOUT_SECONDS:
            logger.info(f"Reusing existing session: {session_id} for dataset_id: {dataset_id}")
            session_data["last_accessed"] = current_time
            return session_data["service"]
        else:
            logger.info(f"Session timed out, removing: {session_id} for dataset_id: {dataset_id}")
            del active_chatbot_sessions[session_key]

    logger.info(f"Creating new service for session: {session_id}, dataset_id: {dataset_id}")
    db_dataset = crud_dataset.get(db, id=dataset_id)
    if not db_dataset:
        # This is a critical error if reached, means dataset_id was not validated before calling this
        logger.error(f"Dataset ID {dataset_id} not found when trying to create service for session {session_id}.")
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found (internal check).")

    if not db_dataset.file_path or not os.path.exists(db_dataset.file_path):
        logger.error(f"Invalid file path for dataset {dataset_id}: {db_dataset.file_path}")
        raise HTTPException(
            status_code=500, # Internal server error as dataset should be valid
            detail=f"File path for dataset ID {dataset_id} is invalid or file does not exist."
        )

    try:
        service = ChatbotService(csv_file_path=db_dataset.file_path)
        active_chatbot_sessions[session_key] = {
            "service": service,
            "last_accessed": current_time
        }
        return service
    except ValueError as ve:
        logger.error(f"ValueError initializing ChatbotService for session {session_id}, dataset {dataset_id}: {ve}")
        raise HTTPException(status_code=500, detail=f"Error initializing chatbot engine: {str(ve)}")
    except Exception as e:
        logger.error(f"Unexpected error initializing ChatbotService for session {session_id}, dataset {dataset_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred setting up the chatbot engine: {e}")

def _cleanup_expired_sessions():
    """Removes timed-out sessions from the in-memory store."""
    current_time = time.time()
    expired_keys = [
        key for key, data in list(active_chatbot_sessions.items()) # list() for safe iteration while deleting
        if (current_time - data["last_accessed"]) >= SESSION_TIMEOUT_SECONDS
    ]
    if expired_keys:
        logger.info(f"Background cleanup: Found {len(expired_keys)} expired sessions.")
        for key in expired_keys:
            logger.info(f"Background cleanup: Removing expired session for dataset_id: {key[0]}, session_id: {key[1]}")
            del active_chatbot_sessions[key]
    else:
        logger.debug("Background cleanup: No expired sessions found.")

# --- Dependency for getting the Chatbot Service with Session ---
class SessionManager:
    def __init__(self, dataset_id: int,
                 session_id_header: Optional[str] = Header(None, alias="X-Session-ID"),
                 session_id_query: Optional[str] = Query(None, alias="session_id")):
        self.dataset_id = dataset_id
        self.session_id_header = session_id_header
        self.session_id_query = session_id_query
        self.resolved_session_id: Optional[str] = None # Will be set

    def _resolve_session_id(self, generate_if_missing: bool = False) -> str:
        if self.resolved_session_id:
            return self.resolved_session_id

        session_id = self.session_id_header or self.session_id_query
        if not session_id and generate_if_missing:
            session_id = str(uuid.uuid4())
            logger.info(f"No session_id provided for dataset {self.dataset_id}, generated new one: {session_id}")
        elif not session_id and not generate_if_missing:
            raise HTTPException(
                status_code=400,
                detail="Session ID is required. Please provide it via 'X-Session-ID' header or 'session_id' query parameter, or start a new session via POST /sessions/start/{dataset_id}."
            )
        self.resolved_session_id = session_id
        return session_id

    async def get_service(self, db: Session = Depends(get_db)) -> ChatbotService:
        """Returns the ChatbotService for the resolved session."""
        session_id = self._resolve_session_id(generate_if_missing=False) # Require session ID for interaction
        
        # Validate dataset existence before trying to get/create service
        db_dataset = crud_dataset.get(db, id=self.dataset_id)
        if not db_dataset:
            raise HTTPException(status_code=404, detail=f"Dataset with ID {self.dataset_id} not found.")

        return _get_or_create_session_service(self.dataset_id, session_id, db)

    async def start_new_session_and_get_service(self, db: Session) -> ChatbotService:
        """Generates a new session ID, creates the service, and returns both."""
        session_id = self._resolve_session_id(generate_if_missing=True) # Generate if missing for new session

        # Validate dataset existence
        db_dataset = crud_dataset.get(db, id=self.dataset_id)
        if not db_dataset:
            raise HTTPException(status_code=404, detail=f"Dataset with ID {self.dataset_id} not found.")

        service = _get_or_create_session_service(self.dataset_id, session_id, db)
        return service, session_id


async def get_service_from_session_manager(
    session_manager: SessionManager = Depends(), # Get the already resolved SessionManager instance
    db: Session = Depends(get_db) # get_service in SessionManager needs the db
) -> ChatbotService:
    # Now call the instance method on the 'session_manager' instance
    return await session_manager.get_service(db=db)

# --- API Endpoints ---

@router.post(
    "/sessions/start/{dataset_id}",
    response_model=chatbot_schemas.NewSessionResponse,
    summary="Start a new chatbot session for a dataset",
    tags=["chatbot_session"]
)
async def start_new_chatbot_session(
    dataset_id: int,
    db: Session = Depends(get_db)
    # No explicit session_id needed here, we generate one
):
    """
    Explicitly starts a new chat session for the given dataset_id.
    A new unique session_id will be generated and returned.
    Use this session_id in subsequent calls to `/interact/{dataset_id}`.
    """
    # Validate dataset first
    db_dataset = crud_dataset.get(db, id=dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found.")

    new_session_id = str(uuid.uuid4())
    logger.info(f"Starting new explicit session {new_session_id} for dataset {dataset_id}")
    _ = _get_or_create_session_service(dataset_id, new_session_id, db) # This creates and stores it

    return chatbot_schemas.NewSessionResponse(
        dataset_id=dataset_id,
        session_id=new_session_id,
        message="New session started. Use the provided session_id for interactions."
    )


@router.post(
    "/interact/{dataset_id}",
    response_model=chatbot_schemas.ChatbotInteractionResponse,
    summary="Interact with the Chatbot for a specific dataset and session",
    tags=["chatbot_interaction"]
)
async def interact_with_chatbot(
    dataset_id: int, # From path
    query_request: chatbot_schemas.ChatbotQueryRequest, # From request body
    background_tasks: BackgroundTasks, # For opportunistic cleanup
    session_manager: SessionManager = Depends(), # Dependency injection
    service: ChatbotService = Depends(get_service_from_session_manager) # Depends on SessionManager instance
):
    """
    Send a query to the chatbot for a given `dataset_id` within a specific session.

    - **dataset_id**: The ID of the dataset to interact with (from path).
    - **X-Session-ID (Header) or session_id (Query Param)**: A unique identifier for the chat session.
      This is REQUIRED for interaction. Use the `/sessions/start/{dataset_id}` endpoint to get one if starting fresh.
    - **Request Body**: Contains the user's `query`.
    """
    # The `service` instance is session-specific due to SessionManager.
    # `session_manager.resolved_session_id` holds the session ID used.
    current_session_id = session_manager.resolved_session_id # Get the session_id used

    if not current_session_id: # Should be caught by SessionManager.get_service, but as a safeguard
        raise HTTPException(status_code=400, detail="Session ID could not be resolved for interaction.")

    try:
        raw_responses = service.process_user_query(user_query=query_request.query)

        formatted_responses: List[chatbot_schemas.ChatbotResponseItem] = []
        for res_item_dict in raw_responses:
            try:
                formatted_responses.append(chatbot_schemas.ChatbotResponseItem(**res_item_dict))
            except Exception as pydantic_error:
                logger.error(f"Pydantic validation error for response item: {res_item_dict}. Error: {pydantic_error}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal server error formatting response for item type '{res_item_dict.get('type', 'Unknown')}'. Error: {pydantic_error}"
                )

        # Opportunistic cleanup
        if random.randint(1, 20) == 1: # Run roughly 1 in 20 requests
            logger.debug(f"Scheduling background session cleanup after interaction for session {current_session_id}")
            background_tasks.add_task(_cleanup_expired_sessions)

        return chatbot_schemas.ChatbotInteractionResponse(
            dataset_id=dataset_id,
            session_id=current_session_id,
            responses=formatted_responses
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during chatbot interaction for dataset {dataset_id}, session {current_session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred during chatbot interaction: {str(e)}")


@router.get(
    "/starter-questions/{dataset_id}",
    response_model=List[str],
    summary="Get starter questions for a dataset (stateless call)",
    tags=["chatbot_utils"]
)
async def get_starter_questions_for_dataset(
    dataset_id: int,
    db: Session = Depends(get_db)
):
    db_dataset = crud_dataset.get(db, id=dataset_id)
    if not db_dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found.")
    if not db_dataset.file_path or not os.path.exists(db_dataset.file_path):
        logger.error(f"Invalid file path for dataset {dataset_id} on starter questions: {db_dataset.file_path}")
        raise HTTPException(status_code=500, detail="Dataset file path invalid or not found.")

    try:
        temp_service = ChatbotService(csv_file_path=db_dataset.file_path)
        return temp_service.get_starter_questions()
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=f"Error initializing temporary chatbot service: {str(ve)}")
    except Exception as e:
        logger.error(f"Error getting starter questions for dataset {dataset_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting starter questions: {str(e)}")


@router.post(
    "/sessions/{dataset_id}/clear-history",
    summary="Clears the chat history and state for a specific session",
    status_code=204,
    tags=["chatbot_session"]
)
async def clear_session_history(
    dataset_id: int,
    session_manager: SessionManager = Depends(), # To resolve session_id
    # service: ChatbotService = Depends(SessionManager.get_service) # Optional: get service to operate on it
    # db: Session = Depends(get_db) # If we need to re-create for some reason
):
    """
    Clears the chat history and resets state (like focus filter, journey log)
    of the ChatbotService instance associated with the given dataset_id and session_id.
    This effectively "restarts" the conversation for that session.
    """
    session_id = session_manager._resolve_session_id(generate_if_missing=False) # Require session ID
    if not session_id: # Should be caught by _resolve_session_id, but defensive
         raise HTTPException(status_code=400, detail="Session ID could not be resolved.")

    session_key = (dataset_id, session_id)
    if session_key in active_chatbot_sessions:
        session_data = active_chatbot_sessions[session_key]
        service_instance = session_data["service"]
        
        # Re-initialize critical state parts of the service instance
        # This is safer than trying to partially clear if the internal state is complex.
        # Or, ChatbotService could have a reset_session_state() method.
        # For now, let's re-initialize the chat session and other state vars.
        try:
            service_instance.chat_session = service_instance._initialize_chat_session() # Re-init chat
            service_instance.analysis_journey_log = []
            service_instance.current_focus_filter = None
            service_instance.pending_code_to_execute = None
            service_instance.pending_whatif_code_to_execute = None
            service_instance.pending_focus_proposal = None
            service_instance.last_executed_plot_path = None
            
            session_data["last_accessed"] = time.time() # Update last accessed time
            logger.info(f"Cleared history and state for session: {session_id}, dataset_id: {dataset_id}")
            return # Returns 204 No Content (FastAPI handles this automatically for None return with 204)
        except Exception as e:
            logger.error(f"Error while resetting state for session {session_id}, dataset {dataset_id}: {e}", exc_info=True)
            # If reset fails, the session might be in an inconsistent state.
            # Consider removing it from active_chatbot_sessions to force a fresh one on next interaction.
            del active_chatbot_sessions[session_key]
            raise HTTPException(status_code=500, detail="Failed to fully clear session state.")
    else:
        logger.info(f"Attempted to clear history for non-existent or timed-out session: {session_id}, dataset_id: {dataset_id}")
        # It's okay if session doesn't exist, "clear" is idempotent.
        return

# Optional: Add an endpoint to explicitly call cleanup if needed for testing/ops
# @router.post("/admin/cleanup-sessions", status_code=204, tags=["admin"])
# async def trigger_session_cleanup():
#     _cleanup_expired_sessions()
#     logger.info("Manual session cleanup triggered.")