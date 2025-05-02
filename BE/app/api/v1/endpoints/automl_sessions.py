from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.orm import Session # Sync Session
from uuid import UUID # Still needed if FinalizedModel uses UUID
from typing import Any # For Any response model if needed later
from app import schemas
from app.services import automl_service
from app.api.v1.dependencies import get_db

# Define the API router
router = APIRouter(
    prefix='/v1',
    tags=['modeling']
)

# --- Endpoint to Start Step 1 ---
@router.post(
    '/automl_sessions', # POST to the base path of this router (e.g., /api/v1/automl-sessions)
    response_model=schemas.AutoMLSessionStep1Response, # Use the specific Step 1 SUCCESS response schema
    status_code=status.HTTP_200_OK, # 200 OK because it's synchronous and returns results
    summary="Start AutoML Step 1: Setup & Compare (Sync & Blocking)",
    responses={ # Define potential error responses using the specific error schema
        status.HTTP_404_NOT_FOUND: {
            "model": schemas.AutoMLSessionErrorResponse,
            "description": "Dataset specified by dataset_id not found in the database."
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": schemas.AutoMLSessionErrorResponse,
            "description": "AutoML Step 1 failed during execution or an internal server error occurred."
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": schemas.AutoMLSessionErrorResponse,
            "description": "Invalid input parameters (e.g., missing target column)."
        },
    }
)
def start_automl_step1_endpoint(
    *,
    db: Session = Depends(get_db), # Inject synchronous DB Session
    params: schemas.AutoMLSessionStartStep1Request = Body(...) # Expect request body matching this schema
):
    """
    Initiates the synchronous AutoML Step 1 (Setup & Compare Models).

    This endpoint performs the following actions sequentially:
    1. Validates the input request body.
    2. Creates an initial record for this AutoML session in the database.
    3. Retrieves the file path associated with the provided `dataset_id`.
    4. Loads the base AutoML configuration (`config.yaml`).
    5. Merges the base configuration with request parameters and application settings.
    6. Instantiates the `AutoMLRunner`.
    7. **BLOCKS** while executing the `step1_setup_and_compare` method of the runner.
    8. Updates the AutoML session record in the database with the final status and results.
    9. Returns the results of Step 1, including the detected task type and comparison results.

    **Request Body:**
    - `dataset_id` (integer, required): ID of the dataset registered in the system.
    - `target_column` (string, required): Name of the column to predict.
    - `feature_columns` (array[string], optional): List of features to use. Uses all others if omitted.
    - `name` (string, optional): Optional descriptive name for this AutoML session.
    """
    try:
        # Call the service function which contains all the logic
        result = automl_service.run_step1_setup_and_compare(db, params)
        # The service function returns the validated AutoMLSessionStep1Response object on success
        return result
    except HTTPException as http_exc:
        # Re-raise HTTPExceptions (like 404 Not Found, 400 Bad Request, 500 Internal Server Error)
        # that might be raised explicitly within the service layer or its dependencies.
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors that weren't converted to HTTPException
        # Log the error for debugging
        print(f"FATAL: Unexpected error in Step 1 endpoint: {e}")
        # Return a generic 500 error to the client
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred during AutoML Step 1: {e}"
        )

@router.post(
    "/{session_id}/step2-tune", # Define a path for step 2, includes session_id
    response_model=schemas.AutoMLSessionStep2Result, # Use the specific Step 2 SUCCESS response
    status_code=status.HTTP_200_OK, # Sync success
    summary="Start AutoML Step 2: Tune & Analyze Model (Sync & Blocking)",
    responses={
        status.HTTP_404_NOT_FOUND: {"model": schemas.AutoMLSessionErrorResponse, "description": "Session or required Step 1 artifacts not found"},
        status.HTTP_400_BAD_REQUEST: {"model": schemas.AutoMLSessionErrorResponse, "description": "Invalid input or prerequisite Step 1 not completed"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": schemas.AutoMLSessionErrorResponse, "description": "AutoML Step 2 failed or Internal Error"}
    }
)
def start_automl_step2_endpoint(
    *,
    session_id: int, # Path parameter for the session
    db: Session = Depends(get_db),
    params: schemas.AutoMLSessionStartStep2Request = Body(...) # Body contains model_id to tune
):
    """
    Initiates the synchronous AutoML Step 2 (Tune & Analyze Model) for a given session.

    - Loads the experiment saved in Step 1.
    - Tunes the specified `model_id_to_tune` based on configuration.
    - Analyzes the tuned model (e.g., plots, SHAP).
    - Saves the tuned model artifact.
    - Updates the Step 2 status and results in the AutoML Session record.
    - **BLOCKS** until Step 2 completes or fails.

    **Path Parameter:**
    - `session_id` (integer): The ID of the AutoML Session created in Step 1.

    **Request Body:**
    - `model_id_to_tune` (string, required): The PyCaret model ID (e.g., 'rf', 'lightgbm') to be tuned. Must be one of the models compared in Step 1.
    """
    try:
        # Call the NEW service function for Step 2
        result = automl_service.run_step2_tune_and_analyze(db, session_id, params)
        return result
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"FATAL: Unexpected error in Step 2 endpoint for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred during AutoML Step 2: {e}"
        )
        
        
@router.post(
    "/{session_id}/step3-finalize", # Define path for step 3
    response_model=schemas.AutoMLSessionStep3Result, # Use Step 3 success response
    status_code=status.HTTP_200_OK, # Sync success
    summary="Start AutoML Step 3: Finalize Model (Sync & Blocking)",
    responses={
        status.HTTP_404_NOT_FOUND: {"model": schemas.AutoMLSessionErrorResponse, "description": "Session or required Step 1/2 artifacts not found"},
        status.HTTP_400_BAD_REQUEST: {"model": schemas.AutoMLSessionErrorResponse, "description": "Invalid input or prerequisite Step 1/2 not completed"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": schemas.AutoMLSessionErrorResponse, "description": "AutoML Step 3 failed or Internal Error"}
    }
)
def start_automl_step3_endpoint(
    *,
    session_id: int, # Path parameter for the session
    db: Session = Depends(get_db),
    params: schemas.AutoMLSessionStartStep3Request = Body(None, description="Optional parameters like model name override.") # Body is optional for this step currently
):
    """
    Initiates the synchronous AutoML Step 3 (Finalize Model).

    - Loads the experiment from Step 1 and the tuned model from Step 2.
    - Finalizes the tuned model (trains on full dataset).
    - Saves the final model artifact.
    - Creates a `FinalizedModel` record in the database.
    - Updates the Step 3 status and results in the AutoML Session record, linking to the `FinalizedModel`.
    - **BLOCKS** until Step 3 completes or fails.

    **Path Parameter:**
    - `session_id` (integer): The ID of the AutoML Session where Steps 1 & 2 completed.

    **Request Body (Optional):**
    - `model_name_override` (string, optional): Provide a specific name for the saved final model artifact.
    """
    try:
        # Use Body(None) if the request body is truly optional,
        # otherwise use Body(...) if some params are expected.
        # If params can be None, handle it in the service layer or use default empty object
        request_params = params if params is not None else schemas.AutoMLSessionStartStep3Request()

        # Call the NEW service function for Step 3
        result = automl_service.run_step3_finalize_and_save(db, session_id, request_params)
        return result
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"FATAL: Unexpected error in Step 3 endpoint for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal error occurred during AutoML Step 3: {e}"
        )