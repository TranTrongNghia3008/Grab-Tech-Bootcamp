# backend/app/services/automl_service.py (Synchronous - Step 1 - New DB Design)
import os
import yaml
import datetime # Needed for timestamp updates
from fastapi import HTTPException
from sqlalchemy.orm import Session # Sync Session
from pydantic import BaseModel
from typing import Dict, Any, Callable, Tuple, Optional
import pandas as pd

# CRUD Imports
from app.crud.automl_sessions import crud_automl_session # Use updated session CRUD
from app.crud.datasets import crud_dataset # Use dataset CRUD
from app.crud.finalized_models import crud_finalized_model, FinalizedModelCreateInternal

# Schema Imports (Make sure schemas package __init__ exports these or import directly)
from app import schemas
# Import specific schemas needed
from app.schemas.automl_sessions import AutoMLSessionStartStep1Request, AutoMLSessionStep1Response, AutoMLSessionStep1Result
from app.schemas.automl_sessions import AutoMLSessionStartStep2Request, AutoMLSessionStep2Result
from app.schemas.automl_sessions import AutoMLSessionStartStep3Request, AutoMLSessionStep3Result
from app.schemas.finalized_models import FinalizedModelResponse # Needed for the result
from app.schemas.commons import DataFrameStructure

# AutoML Runner Import
try:
    from app.automl.automl_copy import AutoMLRunner, load_config, DEFAULT_CONFIG_PATH
except ImportError as e:
    print(f"ERROR: Could not import AutoMLRunner. {e}")
    class AutoMLRunner:
        def __init__(self, config): pass
        def step1_setup_and_compare(self): return None, None, None
    def load_config(path='config.yaml'): return {}
    DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../automl/config.yaml')

# Core Config Import
from app.core.config import settings

# --- Service Function for Step 1 ---
def run_step1_setup_and_compare(db: Session, params: schemas.AutoMLSessionStartStep1Request) -> schemas.AutoMLSessionStep1Response:
    """
    Orchestrates the synchronous execution of AutoML Step 1 with the new DB design:
    1. Creates/Updates AutoML Session DB record to pending status.
    2. Fetches dataset path.
    3. Loads/Merges configuration.
    4. Instantiates AutoMLRunner.
    5. Updates Session status to running.
    6. Executes runner's step1_setup_and_compare (BLOCKING).
    7. Updates Session record with step 1 status, results, paths, run ID.
    8. Returns results.
    """
    # Use the new session CRUD directly
    dataset_crud = crud_dataset

    # 1. Fetch Dataset Path
    dataset = dataset_crud.get(db, id=params.dataset_id)
    if not dataset or not dataset.file_path:
        # Session record doesn't exist yet, so just raise error
        raise HTTPException(status_code=404, detail=f"Dataset with ID {params.dataset_id} not found or has no file path.")
    dataset_file_path = dataset.file_path

    # 2. Create Initial Session Record in DB
    # We use the base create method first
    session_create_data = schemas.AutoMLSessionCreate(
        name=params.name,
        dataset_id=params.dataset_id,
        target_column=params.target_column,
        feature_columns=params.feature_columns
    )
    db_session = db.query(crud_automl_session.model).filter(
        crud_automl_session.model.dataset_id == params.dataset_id
    ).first()
    
    if db_session:
        print(f"Found existing AutoML Session (ID: {db_session.id}) for Dataset ID: {params.dataset_id}. Re-running Step 1.")
        # Reset relevant Step 1 fields before starting
        try:
            db_session = crud_automl_session.update_step_status(
                db, session_id=db_session.id, step_number=1, status="pending",
                results=None, # Clear previous results
                error=None, # Clear previous error
                step1_experiment_path=None, # Clear previous path
                step1_mlflow_run_id=None, # Clear previous run ID
                task_type=None # Clear previous task type
            )
            if not db_session: raise RuntimeError("Failed to reset existing session status.")
        except Exception as reset_e:
            print(f"Database error resetting session {db_session.id} record: {reset_e}")
            raise HTTPException(status_code=500, detail=f"Failed to reset existing AutoML session state: {reset_e}")
    else:
        print(f"No existing session found for Dataset ID: {params.dataset_id}. Creating new session.")
        # Create a new session record if none exists
        session_create_data = schemas.AutoMLSessionCreate(
            name=params.name,
            dataset_id=params.dataset_id,
            target_column=params.target_column,
            feature_columns=params.feature_columns
        )
        try:
            db_session = crud_automl_session.create(db, obj_in=session_create_data)
            db_session = crud_automl_session.update_step_status(db, session_id=db_session.id, step_number=1, status="pending")
            if not db_session: raise RuntimeError("Failed to create or update new session DB record")
            print(f"Created new AutoML Session DB record ID: {db_session.id}, status set to step1_pending.")
        except Exception as db_e:
            print(f"Database error creating session record: {db_e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize AutoML session in database: {db_e}")

    # --- Store session ID for subsequent operations ---
    current_session_id = db_session.id

    # 3. Load base config.yaml
    try:
        base_config = load_config()
    except Exception as cfg_e:
        crud_automl_session.update_step_status(db, session_id=current_session_id, step_number=1, status="failed", error=f"Config Load Error: {cfg_e}")
        crud_automl_session.update_overall_status(db, session_id=current_session_id, status="failed", error=f"Config Load Error: {cfg_e}")
        raise HTTPException(status_code=500, detail=f"Failed to load base AutoML config: {cfg_e}")

    # 4. Update Config In-Memory
    runner_config = base_config.copy()
    runner_config['session_id'] = current_session_id # Use DB ID
    runner_config['data_file_path'] = dataset_file_path
    runner_config['target_column'] = params.target_column
    if params.feature_columns is not None: runner_config['feature_columns'] = params.feature_columns
    if hasattr(settings, 'AUTOML_OUTPUT_BASE_DIR') and settings.AUTOML_OUTPUT_BASE_DIR: runner_config['output_base_dir'] = settings.AUTOML_OUTPUT_BASE_DIR
    if hasattr(settings, 'MLFLOW_TRACKING_URI') and settings.MLFLOW_TRACKING_URI: runner_config['mlflow_tracking_uri'] = settings.MLFLOW_TRACKING_URI

    # Store relevant config in the session's config column
    config_to_store_in_db = {
        "dataset_id": params.dataset_id, "target_column": runner_config['target_column'],
        "feature_columns": runner_config.get('feature_columns'),
        "train_size_used": runner_config.get('sample_fraction', 1.0) if runner_config.get('use_sampling_in_setup') else 1.0,
        "output_base_dir": runner_config.get('output_base_dir'),
        "mlflow_tracking_uri": runner_config.get('mlflow_tracking_uri'),
        "mlflow_experiment_name": runner_config.get('experiment_name')
    }
    # Update config in DB before running
    try:
         crud_automl_session.update_atomic(db, id=current_session_id, values={"config": config_to_store_in_db})
    except Exception as cfg_db_e:
         print(f"Warning: Failed to store runner config in DB session {current_session_id}: {cfg_db_e}")
         # Continue execution, but config won't be persisted

    # 5 & 6. Instantiate Runner and Run Step 1 (Blocking)
    exp_path = None; compare_df = None; detected_task = None; step1_error = None
    final_step1_status = "failed" # Default

    try:
        runner = AutoMLRunner(config=runner_config)
        # Update status to running *using the step-specific method*
        crud_automl_session.update_step_status(db, session_id=current_session_id, step_number=1, status="running")
        print(f"Executing AutoML Step 1 for session {current_session_id}...")

        # --- BLOCKING CALL ---
        exp_path, compare_df, detected_task = runner.step1_setup_and_compare()
        # ---------------------

        print(f"AutoML Step 1 execution finished for session {current_session_id}.")
        if exp_path and detected_task:
            final_step1_status = "completed"
        else:
            step1_error = "AutoML step1 finished but indicated failure (e.g., returned None or empty results)."
            print(step1_error)

    except Exception as runner_e:
        final_step1_status = "failed"
        step1_error = f"Error during AutoMLRunner execution: {type(runner_e).__name__}: {runner_e}"
        print(f"{step1_error}") # Log the exception trace if needed

    # 7. Update Session DB Record with final step 1 status and results
    try:
        # Prepare results dictionary to store in JSON column
        step1_results_dict = {}
        if final_step1_status == "completed":
             step1_results_dict["task_type"] = detected_task
             step1_results_dict["experiment_save_path"] = exp_path
             # Optionally add best model ID or summary of comparison results
             if isinstance(compare_df, pd.DataFrame) and not compare_df.empty:
                  step1_results_dict["best_model_id"] = compare_df.index[0] # Example
                  # step1_results_dict["comparison_summary"] = compare_df.head().to_dict('records') # Example

        # Use the step-specific update method
        update_kwargs = {
            "results": step1_results_dict,
            "error": step1_error, # Pass error message to potentially update session error field
            "task_type": detected_task, # Update the main task_type column
            "step1_experiment_path": exp_path,
            # TODO: Extract mlflow run ID if runner makes it accessible
            # "step1_mlflow_run_id": runner.mlflow_run_id # Assuming runner stores it
        }
        # Remove None values from kwargs before passing to CRUD
        update_kwargs = {k: v for k,v in update_kwargs.items() if v is not None}

        crud_automl_session.update_step_status(
            db,
            session_id=current_session_id,
            step_number=1,
            status=final_step1_status,
            **update_kwargs
        )
        print(f"Updated session {current_session_id} step1 status to {final_step1_status}.")

    except Exception as db_update_e:
        print(f"ERROR: Failed to update final step 1 status/results in DB for {current_session_id}: {db_update_e}")
        raise HTTPException(status_code=500, detail="AutoML step 1 finished but failed to update session status in database.")

    # 8. Prepare and Return Response
    if final_step1_status == "failed":
        raise HTTPException(status_code=500, detail=step1_error or "AutoML Step 1 failed during execution.")

    # Convert pandas DataFrame to DataFrameStructure schema for the response
    comparison_results_schema = None
    if isinstance(compare_df, pd.DataFrame) and not compare_df.empty:
        try:
            df_for_schema = compare_df.reset_index()
            data_list = df_for_schema.astype(object).where(pd.notnull(df_for_schema), None).values.tolist()
            comparison_results_schema = DataFrameStructure(
                columns=df_for_schema.columns.tolist(),
                data=data_list
            )
        except Exception as df_convert_e:
            print(f"Warning: Could not convert comparison results DataFrame to schema: {df_convert_e}")

    # Construct the final success response object
    response_data = {
        "session_id": current_session_id,
        "status": final_step1_status, # Return the step status
        "task_type": detected_task,
        "experiment_save_path": exp_path,
        "comparison_results": comparison_results_schema,
        "error_message": None
    }

    return schemas.AutoMLSessionStep1Response(**response_data)


# --- NEW Service Function for Step 2 ---
def run_step2_tune_and_analyze(db: Session, session_id: int, params: schemas.AutoMLSessionStartStep2Request) -> schemas.AutoMLSessionStep2Result:
    """
    Orchestrates the synchronous execution of AutoML Step 2:
    1. Fetches the existing AutoML Session record.
    2. Validates prerequisites (Step 1 completed, experiment path exists).
    3. Loads base config and merges with session config.
    4. Instantiates AutoMLRunner.
    5. Updates Session Step 2 status to running.
    6. Executes the runner's step2_tune_and_analyze_model method (BLOCKING).
    7. Updates the Session record with step 2 status, results, paths, run ID.
    8. Returns results for Step 2.
    """
    session_crud = crud_automl_session

    # 1. Fetch Existing Session & Validate Prerequisites
    db_session = session_crud.get(db, id=session_id)
    if not db_session:
        raise HTTPException(status_code=404, detail=f"AutoML Session {session_id} not found.")

    # Check if Step 1 completed successfully and necessary artifacts exist
    if db_session.step1_status != "completed" or not db_session.step1_experiment_path or not db_session.task_type:
        raise HTTPException(status_code=400, detail=f"Cannot start Step 2: Step 1 for session {session_id} did not complete successfully or is missing required outputs (experiment path, task type). Current status: {db_session.step1_status}")
    if not os.path.exists(db_session.step1_experiment_path):
         # Update status to reflect the missing artifact issue
         error_msg = f"Cannot start Step 2: Step 1 experiment file not found at '{db_session.step1_experiment_path}'."
         session_crud.update_step_status(db, session_id=session_id, step_number=2, status="failed", error=error_msg)
         session_crud.update_overall_status(db, session_id=session_id, status="step2_failed", error=error_msg)
         raise HTTPException(status_code=404, detail=error_msg)

    # 2. Reset Step 2 status and fields before starting
    try:
        db_session = session_crud.update_step_status(
            db, session_id=session_id, step_number=2, status="pending",
            results=None, error=None, step2_tuned_model_path_base=None,
            step2_mlflow_run_id=None, step2_model_id_tuned=params.model_id_to_tune # Store which model we are tuning now
        )
        if not db_session: raise RuntimeError("Failed to reset session status for Step 2.")
        print(f"Reset session {session_id} for Step 2 execution (tuning '{params.model_id_to_tune}').")
    except Exception as reset_e:
        print(f"Database error resetting session {session_id} for Step 2: {reset_e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset session state for Step 2: {reset_e}")

    # 3. Load base config.yaml and merge
    try:
        base_config = load_config()
        runner_config = base_config.copy()
        # Merge necessary info from the session's stored config
        if db_session.config:
            runner_config.update(db_session.config) # Overlay stored config

        dataset = crud_dataset.get(db, id=db_session.dataset_id)
        
        # Ensure essential IDs and paths are present
        runner_config['session_id'] = session_id
        runner_config['task_type'] = db_session.task_type # Crucial for loading experiment
        runner_config['data_file_path'] = dataset.file_path # Fallback, but should be in config

        # Apply any specific overrides from the request if needed
        # runner_config.update(params.tune_config_overrides or {}) # Example

    except Exception as cfg_e:
        error_msg = f"Config Load/Merge Error for Step 2: {cfg_e}"
        session_crud.update_step_status(db, session_id=session_id, step_number=2, status="failed", error=error_msg)
        session_crud.update_overall_status(db, session_id=session_id, status="step2_failed", error=error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    # 4, 5, 6. Instantiate Runner, Update Status, Run Step 2 (Blocking)
    tuned_model_path_base = None
    step2_error = None
    final_step2_status = "failed" # Default
    step2_results_dict = {} # Store raw results from runner method

    try:
        # Update status to running
        crud_automl_session.update_step_status(db, session_id=session_id, step_number=2, status="running")
        print(f"Executing AutoML Step 2 for session {session_id}...")

        runner = AutoMLRunner(config=runner_config) # Instantiate with merged config

        # --- BLOCKING CALL to Step 2 method ---
        # Assuming step2_tune_and_analyze_model exists in runner and takes experiment_path and model_id
        # NOTE: The placeholder version might need adjustment. This assumes it returns the base path.
        tuned_model_path_base = runner.step2_tune_and_analyze_model(
             experiment_path=db_session.step1_experiment_path,
             model_id_to_tune=params.model_id_to_tune
        )
        # --------------------------------------

        print(f"AutoML Step 2 execution finished for session {session_id}.")
        if tuned_model_path_base:
            final_step2_status = "completed"
            # TODO: Ideally, the runner's step2 method should return more results
            # like best_params, cv_metrics etc. For now, we only have the path.
            step2_results_dict = {
                "best_params": {"placeholder": "fetch from runner/mlflow"},
                "cv_metrics": {"placeholder": "fetch from runner/mlflow"}
            }
        else:
            step2_error = "AutoML step2 finished but indicated failure (tuned model path is None)."
            print(step2_error)

    except Exception as runner_e:
        final_step2_status = "failed"
        step2_error = f"Error during AutoMLRunner Step 2 execution: {type(runner_e).__name__}: {runner_e}"
        print(f"{step2_error}")

    # 7. Update Session DB Record
    try:
        update_kwargs = {
            "results": step2_results_dict, # Store placeholder or real results
            "error": step2_error if final_step2_status == 'failed' else None,
            "step2_tuned_model_path_base": tuned_model_path_base,
            "step2_model_id_tuned": params.model_id_to_tune # Record which model was tuned this run
            # TODO: Extract step 2 mlflow run ID if available
            # "step2_mlflow_run_id": runner.step2_mlflow_run_id # Example
        }
        update_kwargs = {k: v for k,v in update_kwargs.items()} # No need to filter None here? CRUD handles update

        # Use the step-specific update method
        db_session = crud_automl_session.update_step_status(
            db,
            session_id=session_id,
            step_number=2,
            status=final_step2_status,
            **update_kwargs
        )
        if not db_session: raise RuntimeError("Failed to update session status after step 2 execution.")
        print(f"Updated session {session_id} step2 status to {final_step2_status}.")

    except Exception as db_update_e:
        print(f"ERROR: Failed to update final step 2 status/results in DB for {session_id}: {db_update_e}")
        raise HTTPException(status_code=500, detail="AutoML step 2 finished but failed to update session status in database.")

    # 8. Prepare and Return Response
    if final_step2_status == "failed":
        raise HTTPException(status_code=500, detail=step2_error or "AutoML Step 2 failed during execution.")

    # Construct the success response object from step 2 results
    # Map from the step2_results_dict saved in DB if needed
    response_data = {
        "tuned_model_id": step2_results_dict.get("tuned_model_id", params.model_id_to_tune), # Assuming runner doesn't return ID
        "best_params": step2_results_dict.get("best_params"),
        "cv_metrics": step2_results_dict.get("cv_metrics"),
        "tuned_model_save_path_base": tuned_model_path_base,
    }

    return schemas.AutoMLSessionStep2Result(**response_data)

def run_step3_finalize_and_save(db: Session, session_id: int, params: schemas.AutoMLSessionStartStep3Request) -> schemas.AutoMLSessionStep3Result:
    """
    Orchestrates the synchronous execution of AutoML Step 3:
    1. Fetches the existing AutoML Session record.
    2. Validates prerequisites (Step 1&2 completed, experiment/tuned model paths exist).
    3. Loads base config and merges with session config.
    4. Instantiates AutoMLRunner.
    5. Updates Session Step 3 status to running.
    6. Executes runner's step3_finalize_and_save_model method (BLOCKING).
    7. Creates FinalizedModel record in DB.
    8. Updates the Session record linking to FinalizedModel and status/results.
    9. Returns results for Step 3.
    """
    session_crud = crud_automl_session
    finalized_crud = crud_finalized_model # CRUD for FinalizedModel

    # 1. Fetch Existing Session & Validate Prerequisites
    db_session = session_crud.get(db, id=session_id)
    if not db_session:
        raise HTTPException(status_code=404, detail=f"AutoML Session {session_id} not found.")

    # Check if Step 1 & 2 completed successfully and necessary artifacts exist
    if db_session.step1_status != "completed" or not db_session.step1_experiment_path or not db_session.task_type:
        raise HTTPException(status_code=400, detail=f"Cannot start Step 3: Step 1 prerequisites missing for session {session_id}.")
    if db_session.step2_status != "completed" or not db_session.step2_tuned_model_path_base:
         raise HTTPException(status_code=400, detail=f"Cannot start Step 3: Step 2 prerequisites missing for session {session_id}.")

    # Verify artifact paths from previous steps exist
    exp_path = db_session.step1_experiment_path
    tuned_model_path_base = db_session.step2_tuned_model_path_base
    tuned_model_pkl_path = f"{tuned_model_path_base}.pkl" # Construct full path

    if not os.path.exists(exp_path):
        error_msg = f"Cannot start Step 3: Step 1 experiment file not found at '{exp_path}'."
        session_crud.update_step_status(db, session_id=session_id, step_number=3, status="failed", error=error_msg)
        session_crud.update_overall_status(db, session_id=session_id, status="step3_failed", error=error_msg)
        raise HTTPException(status_code=404, detail=error_msg)
    if not os.path.exists(tuned_model_pkl_path):
        error_msg = f"Cannot start Step 3: Step 2 tuned model file not found at '{tuned_model_pkl_path}'."
        session_crud.update_step_status(db, session_id=session_id, step_number=3, status="failed", error=error_msg)
        session_crud.update_overall_status(db, session_id=session_id, status="step3_failed", error=error_msg)
        raise HTTPException(status_code=404, detail=error_msg)

    # 2. Reset Step 3 status and fields before starting
    try:
        db_session = session_crud.update_step_status(
            db, session_id=session_id, step_number=3, status="pending",
            results=None, error=None, step3_final_model_id=None, # Clear previous link
            step3_mlflow_run_id=None
        )
        if not db_session: raise RuntimeError("Failed to reset session status for Step 3.")
        print(f"Reset session {session_id} for Step 3 execution.")
    except Exception as reset_e:
        print(f"Database error resetting session {session_id} for Step 3: {reset_e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset session state for Step 3: {reset_e}")

    # 3. Load base config.yaml and merge (fetch dataset path again for load_experiment)
    try:
        # Re-fetch dataset path needed by load_experiment in automl.py
        dataset = crud_dataset.get(db, id=db_session.dataset_id)
        if not dataset or not dataset.file_path:
             raise ValueError(f"Original Dataset {db_session.dataset_id} path could not be retrieved for finalize step.")
        dataset_file_path = dataset.file_path

        base_config = load_config()
        runner_config = base_config.copy()
        if db_session.config: runner_config.update(db_session.config) # Overlay session config

        # Ensure essentials are set correctly
        runner_config['session_id'] = session_id
        runner_config['task_type'] = db_session.task_type # Use task type from session
        runner_config['data_file_path'] = dataset_file_path # Use fetched path

        # Allow overriding final model name
        if params.model_name_override:
             runner_config['final_model_name_override'] = params.model_name_override # Pass to runner if needed

    except Exception as cfg_e:
        error_msg = f"Config Load/Merge Error for Step 3: {cfg_e}"
        session_crud.update_step_status(db, session_id=session_id, step_number=3, status="failed", error=error_msg)
        session_crud.update_overall_status(db, session_id=session_id, status="step3_failed", error=error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    # 4, 5, 6. Instantiate Runner, Update Status, Run Step 3 (Blocking)
    final_model_path_base = None
    step3_error = None
    final_step3_status = "failed"
    step3_results_dict = {}
    finalized_model_obj = None # To hold the created DB record

    try:
        # Update status to running
        crud_automl_session.update_step_status(db, session_id=session_id, step_number=3, status="running")
        print(f"Executing AutoML Step 3 for session {session_id}...")

        runner = AutoMLRunner(config=runner_config)

        # --- BLOCKING CALL to Step 3 method ---
        # Assuming step3_finalize_and_save_model exists in runner
        final_model_path_base = runner.step3_finalize_and_save_model(
            experiment_path=exp_path,
            tuned_model_path_base=tuned_model_path_base
        )
        # ---------------------------------------

        print(f"AutoML Step 3 execution finished for session {session_id}.")
        if final_model_path_base:
            final_step3_status = "completed"
            # Prepare data to save in FinalizedModel table
            # Use override name if provided, otherwise extract/default
            model_name_to_save = params.model_name_override or os.path.basename(final_model_path_base).split('_')[1] or f"final_model_{session_id}"

            # --- 7. Create FinalizedModel record in DB ---
            fm_create_data = FinalizedModelCreateInternal(
                 session_id=session_id,
                 model_name=model_name_to_save,
                 saved_model_path=f"{final_model_path_base}.pkl",
                 saved_metadata_path=f"{final_model_path_base}_meta.json",
                 # TODO: Extract run ID / URI if runner makes them available
                 # mlflow_run_id=runner.step3_mlflow_run_id,
                 # model_uri_for_registry=runner.step3_model_uri
            )
            finalized_model_obj = finalized_crud.create(db, obj_in=fm_create_data)
            print(f"Created FinalizedModel record ID: {finalized_model_obj.id}")
            step3_results_dict = {
                "finalized_model_db_id": finalized_model_obj.id,
                "saved_model_path": finalized_model_obj.saved_model_path,
                "saved_metadata_path": finalized_model_obj.saved_metadata_path
            }

        else:
            step3_error = "AutoML step3 finished but indicated failure (final model path is None)."
            print(step3_error)

    except Exception as runner_e:
        final_step3_status = "failed"
        step3_error = f"Error during AutoMLRunner Step 3 execution: {type(runner_e).__name__}: {runner_e}"
        print(f"{step3_error}")

    # 8. Update Session DB Record
    try:
        update_kwargs = {
            "results": step3_results_dict, # Store final model details
            "error": step3_error if final_step3_status == 'failed' else None,
            "step3_final_model_id": finalized_model_obj.id if finalized_model_obj else None, # Link to created record
            # TODO: Extract step 3 mlflow run ID if available
            # "step3_mlflow_run_id": runner.step3_mlflow_run_id # Example
        }
        update_kwargs = {k: v for k, v in update_kwargs.items()} # Filter None? Not needed for atomic update

        db_session = crud_automl_session.update_step_status(
            db,
            session_id=session_id,
            step_number=3,
            status=final_step3_status,
            **update_kwargs
        )
        if not db_session: raise RuntimeError("Failed to update session status after step 3 execution.")
        print(f"Updated session {session_id} step3 status to {final_step3_status}.")

    except Exception as db_update_e:
        print(f"ERROR: Failed to update final step 3 status/results in DB for {session_id}: {db_update_e}")
        raise HTTPException(status_code=500, detail="AutoML step 3 finished but failed to update session status in database.")

    # 9. Prepare and Return Response
    if final_step3_status == "failed":
        raise HTTPException(status_code=500, detail=step3_error or "AutoML Step 3 failed during execution.")

    # Construct the success response object from step 3 results
    response_data = {
        "finalized_model_db_id": finalized_model_obj.id if finalized_model_obj else None,
        "saved_model_path": finalized_model_obj.saved_model_path if finalized_model_obj else None,
        "saved_metadata_path": finalized_model_obj.saved_metadata_path if finalized_model_obj else None,
    }

    return schemas.AutoMLSessionStep3Result(**response_data)