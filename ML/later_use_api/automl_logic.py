# automl_logic.py
import pandas as pd
import pycaret.classification as pyclf
import pycaret.regression as pyreg
import time
import numpy as np
import os
import logging
from pathlib import Path

# Import configurations
from config import (
    DEFAULT_UNIQUE_VALUE_THRESHOLD_FOR_CLASSIFICATION,
    CLASSIFICATION_SORT_METRIC, CLASSIFICATION_OPTIMIZE_METRIC,
    REGRESSION_SORT_METRIC, REGRESSION_OPTIMIZE_METRIC,
    DEFAULT_BASELINE_CLASSIFICATION_MODEL, DEFAULT_BASELINE_REGRESSION_MODEL,
    DEFAULT_SAVE_MODEL_DIR, DEFAULT_PLOT_SAVE_DIR, MLFLOW_TRACKING_URI,
    DEFAULT_SESSION_ID
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def auto_detect_task_type(df: pd.DataFrame, target_column: str, unique_threshold: int) -> str:
    """Detects if the task is classification or regression based on the target column."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    target_series = df[target_column]
    n_unique = target_series.nunique()
    dtype = target_series.dtype
    task_type = None

    logger.info(f"Analyzing target column '{target_column}': dtype={dtype}, unique_values={n_unique}")

    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
        task_type = 'classification'
        logger.info("Detected task type: Classification (Target dtype is object/categorical)")
    elif pd.api.types.is_bool_dtype(dtype):
         task_type = 'classification'
         logger.info("Detected task type: Classification (Target dtype is boolean)")
    elif pd.api.types.is_numeric_dtype(dtype):
        unique_values = target_series.unique()
        if len(unique_values) == 2 and np.all(np.isin(unique_values, [0, 1])):
            task_type = 'classification'
            logger.info(f"Detected task type: Binary Classification (Target has values {unique_values})")
        elif n_unique < unique_threshold:
            task_type = 'classification'
            logger.info(f"Detected task type: Classification (Numeric target, {n_unique} unique values < threshold {unique_threshold})")
            # Optional: Warning if float but looks categorical
            if pd.api.types.is_float_dtype(dtype):
                 logger.warning("Target is float but has few unique values. Consider converting to int/category if appropriate.")
        else:
            task_type = 'regression'
            logger.info(f"Detected task type: Regression (Numeric target, {n_unique} unique values >= threshold {unique_threshold})")
    else:
        logger.warning(f"Could not confidently determine task type for target '{target_column}' with dtype {dtype}. Defaulting to classification.")
        task_type = 'classification' # Fallback

    return task_type

def get_pycaret_module_and_metrics(task_type: str):
    """Returns the appropriate PyCaret module and default metrics based on task type."""
    if task_type == 'classification':
        return pyclf, CLASSIFICATION_SORT_METRIC, CLASSIFICATION_OPTIMIZE_METRIC, DEFAULT_BASELINE_CLASSIFICATION_MODEL
    elif task_type == 'regression':
        return pyreg, REGRESSION_SORT_METRIC, REGRESSION_OPTIMIZE_METRIC, DEFAULT_BASELINE_REGRESSION_MODEL
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

# --- Core AutoML Pipeline Function ---

def run_automl_pipeline(
    data_path: str,
    target_column: str,
    session_id: int = DEFAULT_SESSION_ID,
    unique_value_threshold: int = DEFAULT_UNIQUE_VALUE_THRESHOLD_FOR_CLASSIFICATION,
    experiment_name: str = "automl_exp",
    save_model_name: str = "final_automl_model",
    save_plots: bool = True,
    numeric_imputation: str = 'mean',
    categorical_imputation: str = 'mode',
    normalize_regression: bool = True,
    log_experiment_provider: str | None = "mlflow" if MLFLOW_TRACKING_URI else None, # Use mlflow if URI is set
    tracking_uri: str | None = MLFLOW_TRACKING_URI
) -> dict:
    """
    Runs the full PyCaret AutoML pipeline: setup, compare, tune, finalize, save.

    Args:
        data_path: Path to the input CSV data file.
        target_column: Name of the target variable column.
        session_id: Random seed for reproducibility.
        unique_value_threshold: Threshold for distinguishing classification/regression.
        experiment_name: Name for the PyCaret experiment.
        save_model_name: Base name for the saved model file (without .pkl).
        save_plots: Whether to save evaluation plots.
        numeric_imputation: Strategy for numeric imputation.
        categorical_imputation: Strategy for categorical imputation.
        normalize_regression: Whether to normalize features for regression tasks.
        log_experiment_provider: Which experiment tracker to use ('mlflow' or None).
        tracking_uri: The tracking URI if using MLflow.

    Returns:
        A dictionary containing results like task type, best model, metrics, saved model path, etc.
    """
    results = {
        "status": "started",
        "task_type": None,
        "setup_time_s": None,
        "compare_time_s": None,
        "tune_time_s": None,
        "best_model_name": None,
        "best_model_tuned_params": None,
        "baseline_model_name": None,
        "comparison_results": None, # Model comparison metrics
        "final_model_path": None,
        "plot_paths": {},
        "error": None,
    }

    start_pipeline_time = time.time()

    try:
        # --- 1. Load Data ---
        logger.info(f"Loading data from '{data_path}'")
        try:
            data = pd.read_csv(data_path)
            logger.info(f"Data loaded successfully with shape: {data.shape}")
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at '{data_path}'")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")

        # --- 2. Auto-Detect Task Type ---
        logger.info("Auto-detecting task type...")
        results["task_type"] = auto_detect_task_type(data, target_column, unique_value_threshold)
        task_type = results["task_type"]

        pycaret_module, sort_metric, optimize_metric, baseline_model_name_default = get_pycaret_module_and_metrics(task_type)
        logger.info(f"Using PyCaret {task_type} module. Sort/Optimize by: {optimize_metric}")

        # --- 3. Setup PyCaret Environment ---
        logger.info("Setting up PyCaret environment...")
        start_setup_time = time.time()

        setup_args = {
            "data": data,
            "target": target_column,
            "session_id": session_id,
            "log_experiment": log_experiment_provider,
            "experiment_name": experiment_name + f'_{task_type}',
            "numeric_imputation": numeric_imputation,
            "categorical_imputation": categorical_imputation,
            # Add more setup args as needed
        }
        if log_experiment_provider == "mlflow" and tracking_uri:
            setup_args["tracking_uri"] = tracking_uri
            logger.info(f"Logging experiment to MLflow at: {tracking_uri}")
        else:
            logger.info("Experiment logging is disabled or provider not configured.")


        if task_type == 'regression':
            setup_args["normalize"] = normalize_regression

        setup_env = pycaret_module.setup(**setup_args) # Returns the setup tuple
        results["setup_time_s"] = time.time() - start_setup_time
        logger.info(f"PyCaret setup completed in {results['setup_time_s']:.2f} seconds.")

        # --- 4. Compare Models ---
        logger.info("Comparing models...")
        start_compare_time = time.time()
        best_model_obj = pycaret_module.compare_models(sort=sort_metric)
        results_df = pycaret_module.pull()
        results["comparison_results"] = results_df.to_dict(orient='index') # Store metrics
        results["compare_time_s"] = time.time() - start_compare_time
        logger.info(f"Model comparison completed in {results['compare_time_s']:.2f} seconds.")
        logger.info(f"\nModel Comparison Results:\n{results_df}")
        results["best_model_name"] = results_df.index[0] # Get the name of the best model

        # --- 5. Baseline Model Analysis (Optional but good practice) ---
        logger.info("--- Baseline Model Analysis ---")
        baseline_model_name = baseline_model_name_default
        if baseline_model_name not in results_df.index:
            logger.warning(f"Default baseline '{baseline_model_name}' not available. Using best model '{results['best_model_name']}' as baseline reference.")
            baseline_model_name = results['best_model_name'] # Fallback to best if default not found
        else:
            logger.info(f"Selected baseline model for analysis: {baseline_model_name}")

        results["baseline_model_name"] = baseline_model_name
        try:
            baseline_model = pycaret_module.create_model(baseline_model_name, verbose=False) # Less verbose output
            # Save baseline plots if requested
            if save_plots:
                plot_dir = Path(DEFAULT_PLOT_SAVE_DIR)
                plot_dir.mkdir(parents=True, exist_ok=True)
                plot_name_base = f"{save_model_name}_baseline_{baseline_model_name}"
                try:
                     # Evaluate plot (e.g., AUC, Confusion Matrix for clf; Residuals for reg)
                    eval_plot_path = str(plot_dir / f"{plot_name_base}_evaluate")
                    pycaret_module.plot_model(baseline_model, plot='error', save=eval_plot_path, verbose=False) # 'error' often works for both
                    results["plot_paths"][f"baseline_evaluate"] = f"{eval_plot_path}.png"
                    logger.info(f"Saved baseline evaluation plot to {eval_plot_path}.png")
                except Exception as e:
                     logger.warning(f"Could not generate evaluation plot for baseline {baseline_model_name}: {e}")

                try:
                    # Feature Importance
                    feat_plot_path = str(plot_dir / f"{plot_name_base}_feature_importance")
                    pycaret_module.plot_model(baseline_model, plot='feature', save=feat_plot_path, verbose=False)
                    results["plot_paths"]["baseline_feature_importance"] = f"{feat_plot_path}.png"
                    logger.info(f"Saved baseline feature importance plot to {feat_plot_path}.png")
                except Exception as e:
                    logger.warning(f"Could not generate feature importance plot for baseline model {baseline_model_name}: {e}")

        except Exception as e:
            logger.error(f"Error during baseline model ('{baseline_model_name}') analysis: {e}")


        # --- 6. Complex Model Tuning & Analysis (Best Model) ---
        logger.info(f"--- Tuning Best Model: {results['best_model_name']} ---")
        start_tune_time = time.time()
        # Use the actual best model object from compare_models
        tuned_best_model = pycaret_module.tune_model(best_model_obj, optimize=optimize_metric, verbose=False)
        results["tune_time_s"] = time.time() - start_tune_time
        logger.info(f"Model tuning completed in {results['tune_time_s']:.2f} seconds.")
        # Extract tuned parameters (can be complex for pipelines)
        try:
            # Simple extraction for common estimators, might need refinement for complex pipelines
            if hasattr(tuned_best_model, 'get_params'):
                 results["best_model_tuned_params"] = str(tuned_best_model.get_params()) # Convert params dict to string for JSON
            else:
                 results["best_model_tuned_params"] = "Could not extract parameters"
        except Exception:
            results["best_model_tuned_params"] = "Could not extract parameters (exception)"

        logger.info(f"Tuned Model Parameters: {results['best_model_tuned_params']}")

        # Save tuned model plots if requested
        if save_plots:
            plot_dir = Path(DEFAULT_PLOT_SAVE_DIR)
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_name_base = f"{save_model_name}_tuned_{results['best_model_name']}"
            try:
                eval_plot_path = str(plot_dir / f"{plot_name_base}_evaluate")
                pycaret_module.plot_model(tuned_best_model, plot='error', save=eval_plot_path, verbose=False)
                results["plot_paths"][f"tuned_evaluate"] = f"{eval_plot_path}.png"
                logger.info(f"Saved tuned model evaluation plot to {eval_plot_path}.png")
            except Exception as e:
                logger.warning(f"Could not generate evaluation plot for tuned model: {e}")

            try:
                feat_plot_path = str(plot_dir / f"{plot_name_base}_feature_importance")
                pycaret_module.plot_model(tuned_best_model, plot='feature', save=feat_plot_path, verbose=False)
                results["plot_paths"]["tuned_feature_importance"] = f"{feat_plot_path}.png"
                logger.info(f"Saved tuned model feature importance plot to {feat_plot_path}.png")
            except Exception as e:
                logger.warning(f"Could not generate feature importance plot for tuned model: {e}")

        # --- 7. Finalize and Save Model ---
        logger.info("Finalizing model (training on full dataset)...")
        final_model = pycaret_module.finalize_model(tuned_best_model)

        model_save_path = Path(DEFAULT_SAVE_MODEL_DIR) / f"{save_model_name}.pkl"
        model_save_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        pycaret_module.save_model(final_model, str(model_save_path.with_suffix(''))) # save_model adds .pkl
        results["final_model_path"] = str(model_save_path)
        logger.info(f"Final pipeline saved as '{results['final_model_path']}'")

        results["status"] = "completed"
        logger.info("--- AutoML Pipeline Completed Successfully ---")

    except (ValueError, FileNotFoundError, RuntimeError) as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        results["status"] = "failed"
        results["error"] = str(e)
    except Exception as e:
        logger.error(f"An unexpected error occurred during the pipeline: {e}", exc_info=True)
        results["status"] = "failed"
        results["error"] = f"An unexpected error occurred: {e}"

    results["total_time_s"] = time.time() - start_pipeline_time
    logger.info(f"Total pipeline execution time: {results['total_time_s']:.2f} seconds.")
    return results


# --- Prediction Function ---

def load_model_and_predict(model_name: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Loads a saved PyCaret model and makes predictions on new data.

    Args:
        model_name: The name of the model file (e.g., 'my_model.pkl') located
                    in the DEFAULT_SAVE_MODEL_DIR.
        data: A pandas DataFrame containing the new data for prediction.
              Column names must match the training data (excluding the target).

    Returns:
        A pandas DataFrame with predictions added.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
        Exception: For any other prediction errors.
    """
    model_path = Path(DEFAULT_SAVE_MODEL_DIR) / model_name
    model_path_base = str(model_path.with_suffix('')) # Pycaret load_model needs path without extension

    if not model_path.exists():
        logger.error(f"Model file not found at '{model_path}'")
        raise FileNotFoundError(f"Model file '{model_name}' not found in '{DEFAULT_SAVE_MODEL_DIR}'.")

    try:
        logger.info(f"Loading model from '{model_path}'...")
        # PyCaret's load_model implicitly knows the correct module (clf/reg)
        loaded_pipeline = pyclf.load_model(model_path_base, verbose=False) # Try clf first
        if loaded_pipeline is None: # If clf fails, try reg
             loaded_pipeline = pyreg.load_model(model_path_base, verbose=False)

        if loaded_pipeline is None:
             raise RuntimeError(f"Failed to load model '{model_name}' using either classification or regression modules.")

        logger.info(f"Model '{model_name}' loaded successfully.")

        # Determine the correct predict_model function
        pycaret_module = pyclf if hasattr(loaded_pipeline, 'predict_proba') else pyreg

        logger.info(f"Making predictions with {pycaret_module.__name__}...")
        predictions_df = pycaret_module.predict_model(loaded_pipeline, data=data)
        logger.info("Predictions generated successfully.")
        return predictions_df

    except Exception as e:
        logger.error(f"An error occurred during prediction with model '{model_name}': {e}", exc_info=True)
        # Re-raise the exception to be caught by the API endpoint
        raise RuntimeError(f"Prediction failed for model '{model_name}': {e}")