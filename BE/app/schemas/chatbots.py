# app/schemas/chatbot.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union

class ChatbotQueryRequest(BaseModel):
    query: str = Field(..., description="The user's query or command for the chatbot.")
    # session_id: Optional[str] = None # We'll primarily use Header/Query for session_id in API path

class ChatbotResponseItem(BaseModel):
    type: str = Field(..., description="Type of the response item.")
    message: Optional[str] = None
    content: Optional[Union[str, List[str], Dict[str, Any]]] = None

    # Code related
    code: Optional[str] = None
    code_attempted: Optional[str] = None
    is_plot_code: Optional[bool] = None

    # File/Path related
    path: Optional[str] = None

    # Execution/Output related
    output: Optional[str] = None
    last_error: Optional[str] = None

    # AI Interaction specific
    explanation: Optional[str] = None
    insights: Optional[str] = None
    questions: Optional[List[str]] = None
    ai_review: Optional[str] = None

    # Journey/History
    log: Optional[List[Dict[str, Any]]] = None

    # UI/UX flow
    next_actions: Optional[List[str]] = None
    attempt_number: Optional[int] = None

    # Contextual info
    original_query: Optional[str] = None
    filter_condition: Optional[str] = None

    # /focus specific
    impact_assessment: Optional[str] = None

    class Config:
        # from_attributes = True # For Pydantic V2
        pass

class ChatbotInteractionResponse(BaseModel):
    dataset_id: int
    session_id: str # Now required, server will ensure it's set
    responses: List[ChatbotResponseItem]

class NewSessionResponse(BaseModel):
    dataset_id: int
    session_id: str
    message: str