# backend/app/schemas/__init__.py

# --- Import from common.py ---
# Ensure you have a schemas/common.py file with these
try:
    from .commons import StatusResponse, DataFrameStructure
except ImportError:
    print("Warning: Could not import schemas from common.py")
    # Define dummies if needed for other imports to work initially
    class StatusResponse: pass
    class DataFrameStructure: pass


# --- Import from automl_session.py ---
# Import all relevant schemas defined in that file
try:
    from .automl_sessions import (
        AutoMLSessionBase,
        AutoMLSessionCreate,
        AutoMLSessionStartStep1Request, # Schema for starting step 1
        AutoMLSessionStep1Response,
        AutoMLSessionStep1Result,       # Schema for step 1 results (used in Response)
        AutoMLSessionStartStep2Request, # Schema for starting step 2
        AutoMLSessionStep2Result,       # Schema for step 2 results
        AutoMLSessionStartStep3Request, # Schema for starting step 3
        AutoMLSessionStep3Result,       # Schema for step 3 results
        AutoMLSessionResponse,          # The main response schema showing all steps
        AutoMLSessionErrorResponse,
        AutoMLSessionUpdateStepStatus   # Internal schema for updates (optional export)
    )
except ImportError:
    print("Warning: Could not import schemas from automl_session.py")


# --- Import from finalized_models.py ---
# Import schemas related to the FinalizedModel table
try:
    from .finalized_models import (
        FinalizedModelResponse
        # Add FinalizedModelCreate/Update schemas here if you defined them
    )
except ImportError:
     print("Warning: Could not import schemas from finalized_models.py")


# --- Import from other necessary schema files ---
# (Uncomment and add imports as you create these files/schemas)

# from .model_comparison import CompareModelsRequest, CompareModelsResult
# from .model_tuning import TuneModelRequest, TuneModelResult
# from .ensemble import CreateEnsembleRequest, EnsembleResult
# from .analysis import AnalyzeModelRequest, AnalysisResult
# from .finalize import FinalizeModelRequest, FinalizeResult # Note: Step 3 Request/Result are in automl_session.py now
# from .registry import RegisterModelRequest, RegisterModelResponse
# from .drift import CheckDriftRequest, DriftResult
# from .report import GenerateReportResponse
# from .explain import ExplainPredictionRequest, ExplainResult
# from .predict import PredictionRequest, PredictionResult
# from .artifact import CodeSnippetResponse


# --- Optional: Define __all__ ---
# This explicitly lists what is available when using 'from app.schemas import *'
# Helps with code analysis and prevents accidental imports.
# __all__ = [
#     # Common
#     "StatusResponse", "DataFrameStructure",
#     # Session
#     "AutoMLSessionBase", "AutoMLSessionCreate", "AutoMLSessionStartStep1Request",
#     "AutoMLSessionStep1Result", "AutoMLSessionStartStep2Request", "AutoMLSessionStep2Result",
#     "AutoMLSessionStartStep3Request", "AutoMLSessionStep3Result", "AutoMLSessionResponse",
#     "AutoMLSessionErrorResponse", "AutoMLSessionUpdateStepStatus",
#     # Finalized Model
#     "FinalizedModelResponse",
#     # Add others as you import them above...
# ]
