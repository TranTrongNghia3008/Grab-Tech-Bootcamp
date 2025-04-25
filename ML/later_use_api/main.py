# main.py
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
import pandas as pd
import logging

# Import logic functions and schemas
from automl_logic import run_automl_pipeline, load_model_and_predict, logger
from schemas import TrainRequest, TrainResponse, PredictRequest, PredictResponse
from config import DEFAULT_SAVE_MODEL_DIR # To confirm model existence

app = FastAPI(
    title="AutoML API",
    description="API for running AutoML pipelines and making predictions using PyCaret.",
    version="1.0.0",
)

# --- API Endpoints ---

@app.post("/train", response_model=TrainResponse, tags=["Training"])
async def train_automl_model(request: TrainRequest = Body(...)):
    """
    Triggers the AutoML training pipeline based on the provided configuration.

    **Note:** This is a synchronous endpoint. For long-running training jobs,
    consider implementing asynchronous tasks (e.g., using Celery or BackgroundTasks).
    """
    logger.info(f"Received training request: {request.dict(exclude={'data_path'})}") # Log request details safely

    # Basic check for data path existence (server-side check)
    # In a real scenario, you might have more robust checks or ways to access data
    # like S3 paths, database connections etc.
    # For now, we assume the path is accessible by the server.
    # import os
    # if not os.path.exists(request.data_path):
    #     logger.error(f"Data path not found on server: {request.data_path}")
    #     raise HTTPException(status_code=400, detail=f"Data file not found at path: {request.data_path}")

    try:
        # Run the pipeline
        results = run_automl_pipeline(
            data_path=request.data_path,
            target_column=request.target_column,
            session_id=request.session_id,
            unique_value_threshold=request.unique_value_threshold,
            experiment_name=request.experiment_name,
            save_model_name=request.save_model_name,
            save_plots=request.save_plots,
            numeric_imputation=request.numeric_imputation,
            categorical_imputation=request.categorical_imputation,
            normalize_regression=request.normalize_regression,
            log_experiment_provider=request.log_experiment_provider,
            # tracking_uri= # Pass tracking URI if needed, maybe from config
        )

        # Determine appropriate status code based on results
        if results["status"] == "failed":
            # Return 500 for internal errors during processing
            return JSONResponse(status_code=500, content=results)
        elif results["status"] == "completed":
            # Return 200 for successful completion
             return JSONResponse(status_code=200, content=results)
        else:
             # Should not happen if logic is correct, but handle just in case
             return JSONResponse(status_code=500, content=results)


    except FileNotFoundError as e:
        logger.error(f"Training failed - File Not Found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Training failed - Value Error: {e}")
        raise HTTPException(status_code=400, detail=str(e)) # Bad request (e.g., bad column name)
    except RuntimeError as e:
        logger.error(f"Training failed - Runtime Error: {e}")
        raise HTTPException(status_code=500, detail=f"Training pipeline runtime error: {e}")
    except Exception as e:
        logger.exception("An unexpected error occurred during training.") # Log full traceback
        raise HTTPException(status_code=500, detail=f"An unexpected internal server error occurred: {e}")


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_with_model(request: PredictRequest = Body(...)):
    """
    Makes predictions using a previously trained and saved AutoML model.
    """
    logger.info(f"Received prediction request for model: {request.model_name}")

    if not request.data:
         raise HTTPException(status_code=400, detail="No data provided for prediction.")

    try:
        # Convert incoming data (list of dicts) to DataFrame
        input_df = pd.DataFrame(request.data)
        logger.info(f"Prediction input data shape: {input_df.shape}")

        # Call the prediction logic
        predictions_df = load_model_and_predict(request.model_name, input_df)

        # Convert predictions DataFrame back to list of dictionaries for JSON response
        # Keep only relevant columns if needed, predict_model adds prediction_label/score
        predictions_list = predictions_df.to_dict(orient='records')

        return PredictResponse(
            model_used=request.model_name,
            predictions=predictions_list
        )

    except FileNotFoundError as e:
        logger.error(f"Prediction failed - Model Not Found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, RuntimeError) as e: # Catch errors from prediction logic
         logger.error(f"Prediction failed: {e}")
         raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("An unexpected error occurred during prediction.")
        raise HTTPException(status_code=500, detail=f"An unexpected internal server error occurred during prediction: {e}")

@app.get("/", tags=["General"])
async def read_root():
    """Root endpoint providing basic API information."""
    return {"message": "Welcome to the AutoML API. Visit /docs for documentation."}

# --- Optional: Add endpoint to list available models ---
@app.get("/models", tags=["Prediction"])
async def list_available_models():
    """Lists models available in the save directory."""
    from pathlib import Path
    model_dir = Path(DEFAULT_SAVE_MODEL_DIR)
    try:
        models = [f.name for f in model_dir.glob("*.pkl")]
        return {"available_models": models}
    except Exception as e:
        logger.error(f"Error listing models in {model_dir}: {e}")
        raise HTTPException(status_code=500, detail="Could not list available models.")


# --- How to run (using uvicorn) ---
# Save this file as main.py
# Run from terminal: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# --reload: automatically reloads server on code changes (for development)
# --host 0.0.0.0: makes the API accessible on your network
# --port 8000: specifies the port number