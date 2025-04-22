# schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- Request Models ---

class TrainRequest(BaseModel):
    data_path: str = Field(..., description="Path to the input CSV data file on the server.")
    target_column: str = Field(..., description="Name of the target variable column.")
    session_id: int = Field(DEFAULT_SESSION_ID, description="Random seed for reproducibility.")
    unique_value_threshold: int = Field(DEFAULT_UNIQUE_VALUE_THRESHOLD_FOR_CLASSIFICATION, description="Threshold of unique target values to distinguish classification from regression.")
    experiment_name: str = Field(DEFAULT_EXPERIMENT_NAME_PREFIX, description="Base name for the PyCaret experiment.")
    save_model_name: str = Field("final_automl_model", description="Base name for the saved model file (without .pkl).")
    save_plots: bool = Field(True, description="Whether to save evaluation plots.")
    # Add other tunable parameters from run_automl_pipeline if needed
    numeric_imputation: str = Field('mean', description="Numeric imputation strategy ('mean', 'median', 'knn', etc.).")
    categorical_imputation: str = Field('mode', description="Categorical imputation strategy ('mode', 'knn', etc.).")
    normalize_regression: bool = Field(True, description="Apply normalization in regression tasks.")
    log_experiment_provider: Optional[str] = Field(None, description="Experiment tracker ('mlflow' or None). Requires server-side setup if used.")


class PredictRequest(BaseModel):
    model_name: str = Field(..., description="Name of the saved model file (e.g., 'final_automl_model.pkl').")
    # Data can be a list of dictionaries, where each dict is a row
    data: List[Dict[str, Any]] = Field(..., description="List of records (dictionaries) for prediction. Keys must match feature names used during training.")

# --- Response Models ---

class TrainResponse(BaseModel):
    status: str = Field(..., description="Status of the training job ('started', 'completed', 'failed').")
    task_type: Optional[str] = Field(None, description="Detected task type ('classification' or 'regression').")
    setup_time_s: Optional[float] = Field(None, description="Time taken for PyCaret setup.")
    compare_time_s: Optional[float] = Field(None, description="Time taken for model comparison.")
    tune_time_s: Optional[float] = Field(None, description="Time taken for tuning the best model.")
    total_time_s: Optional[float] = Field(None, description="Total pipeline execution time.")
    best_model_name: Optional[str] = Field(None, description="Name/ID of the best performing model found.")
    best_model_tuned_params: Optional[str] = Field(None, description="Parameters of the tuned best model (as string).")
    baseline_model_name: Optional[str] = Field(None, description="Name/ID of the baseline model analyzed.")
    comparison_results: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Metrics from model comparison (model_name -> {metric: value}).")
    final_model_path: Optional[str] = Field(None, description="Path where the final trained model pipeline was saved.")
    plot_paths: Dict[str, str] = Field({}, description="Dictionary of saved plot names to their server paths.")
    error: Optional[str] = Field(None, description="Error message if the job failed.")

class PredictResponse(BaseModel):
    model_used: str
    predictions: List[Dict[str, Any]] = Field(..., description="List of input records with prediction results added.")
    error: Optional[str] = None