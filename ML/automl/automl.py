import pandas as pd
import pycaret.classification as pyclf
import pycaret.regression as pyreg
import time
import numpy as np
import os
import joblib
import mlflow
import datetime
import logging
from typing import List, Optional, Dict, Any, Tuple
import pycaret # Keep for version check if needed elsewhere

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Class (Example using simple dictionary) ---
# For larger projects, consider using Pydantic, dataclasses, or loading from YAML/JSON
CONFIG = {
    "data_file_path": 'C:\Math-Students.csv',
    "target_column": 'G1',
    "unique_value_threshold_for_classification": 20,
    "session_id": 123,
    "baseline_classification_models": ['lr', 'rf'],
    "baseline_regression_models": ['lr', 'rf'],
    "experiment_name": 'automl_autodetect_exp_refactored',
    "model_save_dir": './automl_models_refactored',
    "mlflow_tracking_uri": "sqlite:///mlflow_refactored.db",
    # --- PyCaret Setup Params ---
    "numeric_imputation": 'mean',
    "categorical_imputation": 'mode',
    "normalize_regression": True,
    # --- PyCaret Model Selection/Tuning Params ---
    "sort_metric_classification": 'Accuracy',
    "optimize_metric_classification": 'Accuracy',
    "sort_metric_regression": 'R2',
    "optimize_metric_regression": 'R2',
    "tuning_folds": 5,
    "baseline_folds": 5,
    "tuning_search_library": 'scikit-learn',
    "tuning_search_algorithm": 'random', # 'random' or 'grid'
    "tuning_iterations": 20 # Used for 'random' search
}

class AutoMLRunner:
    """
    Encapsulates the AutoML workflow using PyCaret and MLflow.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the AutoMLRunner with configuration.

        Args:
            config: A dictionary containing all necessary configuration parameters.
        """
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.task_type: Optional[str] = None
        self.pycaret_module: Optional[Any] = None
        self.sort_metric: Optional[str] = None
        self.optimize_metric: Optional[str] = None
        self.baseline_model_ids: List[str] = []
        self.setup_env: Optional[Any] = None # Stores the result of pycaret.setup
        self.best_model_obj: Optional[Any] = None # Best model from compare_models
        self.results_df: Optional[pd.DataFrame] = None # Results from compare_models
        self.baseline_models_trained: Dict[str, Any] = {} # Store trained baselines
        self.tuned_best_model: Optional[Any] = None # Tuned best model object
        self.final_model: Optional[Any] = None # Final model trained on full data

        os.makedirs(self.config["model_save_dir"], exist_ok=True)
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Sets up MLflow tracking URI and experiment."""
        logging.info("--- Setting up MLflow Tracking ---")
        mlflow.set_tracking_uri(self.config["mlflow_tracking_uri"])
        mlflow.set_experiment(self.config["experiment_name"])
        logging.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        logging.info(f"MLflow experiment set to: {self.config['experiment_name']}")

    def load_data(self) -> bool:
        """Loads data from the specified file path."""
        logging.info("--- Loading Data ---")
        try:
            self.data = pd.read_csv(self.config["data_file_path"])
            logging.info(f"Data loaded successfully from '{self.config['data_file_path']}' with shape: {self.data.shape}")
            if self.config["target_column"] not in self.data.columns:
                raise ValueError(f"Target column '{self.config['target_column']}' not found.")
            # Reset index if needed, can prevent issues with some PyCaret versions/data
            self.data = self.data.reset_index(drop=True)
            return True
        except FileNotFoundError:
            logging.error(f"Data file not found at '{self.config['data_file_path']}'")
            return False
        except Exception as e:
            logging.error(f"Error loading data: {e}", exc_info=True)
            return False

    def detect_task_type(self) -> bool:
        """Automatically detects the task type (classification/regression)."""
        if self.data is None:
            logging.error("Data not loaded. Cannot detect task type.")
            return False

        logging.info("--- Auto-Detecting Task Type ---")
        target_series = self.data[self.config["target_column"]]
        n_unique = target_series.nunique()
        dtype = target_series.dtype

        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
            self.task_type = 'classification'
            logging.info("Detected task type: Classification (Target dtype is object/category)")
        elif pd.api.types.is_bool_dtype(dtype):
             self.task_type = 'classification'
             logging.info("Detected task type: Classification (Target dtype is bool)")
        elif pd.api.types.is_numeric_dtype(dtype):
            unique_values = target_series.unique()
            # Explicit check for binary integers {0, 1}
            if len(unique_values) == 2 and np.all(np.isin(unique_values, [0, 1])):
                self.task_type = 'classification'
                logging.info(f"Detected task type: Binary Classification (Target has values {unique_values})")
            # Check unique value count threshold
            elif n_unique < self.config["unique_value_threshold_for_classification"]:
                self.task_type = 'classification'
                logging.info(f"Detected task type: Classification (Target dtype {dtype}, {n_unique} unique < threshold)")
                # Optional: Check if float target with few unique values should be treated as classification
                if pd.api.types.is_float_dtype(dtype):
                     logging.warning("Target is float but has few unique values. Treating as classification.")
            else:
                self.task_type = 'regression'
                logging.info(f"Detected task type: Regression (Target dtype {dtype}, {n_unique} unique >= threshold)")
        else:
            logging.warning(f"Could not reliably determine task type for dtype {dtype}. Defaulting to classification.")
            self.task_type = 'classification' # Default fallback

        return True


    def setup_pycaret(self) -> bool:
        """Sets up the PyCaret environment based on the detected task type."""
        if self.data is None or self.task_type is None:
            logging.error("Data or task type not available for PyCaret setup.")
            return False

        logging.info("--- Setting up PyCaret Environment ---")
        start_time = time.time()

        if self.task_type == 'classification':
            self.pycaret_module = pyclf
            self.sort_metric = self.config["sort_metric_classification"]
            self.optimize_metric = self.config["optimize_metric_classification"]
            self.baseline_model_ids = self.config["baseline_classification_models"]
            logging.info(f"Using PyCaret Classification. Sort/Optimize Metric: {self.sort_metric}")

            setup_params = {
                "data": self.data,
                "target": self.config["target_column"],
                "session_id": self.config["session_id"],
                "log_experiment": True,
                "experiment_name": self.config["experiment_name"],
                "numeric_imputation": self.config["numeric_imputation"],
                "categorical_imputation": self.config["categorical_imputation"]
            }
            # Add specific classification params if needed

        elif self.task_type == 'regression':
            self.pycaret_module = pyreg
            self.sort_metric = self.config["sort_metric_regression"]
            self.optimize_metric = self.config["optimize_metric_regression"]
            self.baseline_model_ids = self.config["baseline_regression_models"]
            logging.info(f"Using PyCaret Regression. Sort/Optimize Metric: {self.sort_metric}")

            setup_params = {
                "data": self.data,
                "target": self.config["target_column"],
                "session_id": self.config["session_id"],
                "log_experiment": True,
                "experiment_name": self.config["experiment_name"],
                "numeric_imputation": self.config["numeric_imputation"],
                "normalize": self.config["normalize_regression"]
            }
            # Add specific regression params if needed
        else:
            logging.error(f"Invalid task type determined: {self.task_type}")
            return False

        try:
            # We assume we are already inside an active MLflow run context
            self.setup_env = self.pycaret_module.setup(**setup_params)
            logging.info(f"PyCaret setup completed in {time.time() - start_time:.2f} seconds.")
             # Log setup metrics/params that PyCaret might not log automatically
            mlflow.log_param("pycaret_sort_metric", self.sort_metric)
            mlflow.log_param("pycaret_optimize_metric", self.optimize_metric)
            return True
        except Exception as e:
            logging.error(f"PyCaret setup failed: {e}", exc_info=True)
            return False

    def compare_models(self) -> bool:
        """Compares baseline models using PyCaret."""
        if self.pycaret_module is None or self.sort_metric is None:
            logging.error("PyCaret environment not set up for model comparison.")
            return False

        logging.info("--- Comparing Models ---")
        start_time = time.time()
        try:
            # You might want to include/exclude specific models here
            self.best_model_obj = self.pycaret_module.compare_models(sort=self.sort_metric)
            self.results_df = self.pycaret_module.pull() # Get results grid
            logging.info("\nModel Comparison Results (Cross-Validated):")
            # Log dataframe in a more readable format if possible, or just head/tail
            print(self.results_df.head()) # Print head for console readability
            logging.info(f"Model comparison completed in {time.time() - start_time:.2f} seconds.")

            # Log results table to MLflow
            if self.results_df is not None:
                results_html = self.results_df.to_html(escape=False)
                mlflow.log_text(results_html, "compare_models_results.html")
            return True
        except Exception as e:
            logging.error(f"Model comparison failed: {e}", exc_info=True)
            return False

    def _log_plot_artifact(self, plot_function: callable, plot_name: str, **kwargs):
        """Helper to generate, save, and log a plot artifact."""
        logging.info(f"Generating '{plot_name}' plot...")
        plot_save_path = None
        try:
            # Call the provided plot function (plot_model or interpret_model)
            plot_save_path = plot_function(save=True, verbose=False, **kwargs)
            logging.info(f"Plot saved locally to: {plot_save_path}")

            if plot_save_path and os.path.exists(plot_save_path):
                try:
                    mlflow.log_artifact(plot_save_path)
                    logging.info(f"Successfully logged plot artifact: {plot_save_path}")
                except Exception as log_e:
                    logging.error(f"Could not log plot artifact {plot_save_path}: {log_e}", exc_info=True)
            else:
                logging.warning(f"Plot file not found or path invalid for '{plot_name}'. Expected path: {plot_save_path}")
        except Exception as plot_e:
            logging.error(f"Failed to generate or save plot '{plot_name}': {plot_e}", exc_info=True)


    def _save_model_artifact(self, model_object: Any, model_stage: str, model_id: str) -> Optional[str]:
        """Helper to save a model artifact with timestamp and log path."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename_base = f"{model_stage}_{model_id}_{timestamp}"
            save_path_base = os.path.join(self.config["model_save_dir"], model_filename_base)

            self.pycaret_module.save_model(model_object, save_path_base)
            full_save_path = f"{save_path_base}.pkl" # save_model adds .pkl
            logging.info(f"Model '{model_id}' ({model_stage}) saved to {full_save_path}")
            # Optional: Log the saved model path as an MLflow parameter/tag
            mlflow.log_param(f"{model_stage}_{model_id}_save_path", full_save_path)
            # Note: PyCaret's log_experiment=True might automatically log the model artifact itself
            return full_save_path
        except Exception as e:
            logging.error(f"Failed to save model artifact {model_stage} {model_id}: {e}", exc_info=True)
            return None

    def analyze_baseline_models(self) -> bool:
        """Creates, evaluates, logs plots, and saves specified baseline models."""
        if not self.baseline_model_ids or self.pycaret_module is None:
             logging.error("Baseline models or PyCaret module not defined.")
             return False

        logging.info("--- Baseline Model Analysis ---")
        success = True
        for baseline_id in self.baseline_model_ids:
            logging.info(f"\nAnalyzing baseline model: {baseline_id}")
            try:
                # Create baseline model (trained on CV folds)
                baseline_model = self.pycaret_module.create_model(
                    baseline_id, fold=self.config["baseline_folds"]
                )
                self.baseline_models_trained[baseline_id] = baseline_model

                logging.info(f"Evaluating Baseline Model: {baseline_id}")
                # Evaluate on hold-out set (displays plots if run interactively)
                _ = self.pycaret_module.evaluate_model(baseline_model)

                # Log feature importance plot
                self._log_plot_artifact(
                    plot_function=lambda save, verbose: self.pycaret_module.plot_model(baseline_model, plot='feature', save=save, verbose=verbose),
                    plot_name=f"Feature Importance (Baseline {baseline_id})"
                 )

                # Save baseline model artifact
                self._save_model_artifact(baseline_model, "baseline", baseline_id)

            except Exception as e:
                logging.error(f"Error processing baseline model '{baseline_id}': {e}", exc_info=True)
                success = False # Mark overall analysis as potentially incomplete
                continue # Continue to the next baseline model

        return success


    def tune_best_model(self) -> bool:
        """Tunes the best model found by compare_models."""
        if self.best_model_obj is None or self.pycaret_module is None:
            logging.error("Best model or PyCaret module not available for tuning.")
            return False
        if self.results_df is None or self.results_df.empty:
             logging.warning("Results dataframe is empty, cannot get best model ID.")
             best_model_id = "unknown_best" # Fallback ID
        else:
            best_model_id = self.results_df.index[0]

        logging.info(f"--- Tuning Best Model: {best_model_id} ---")
        start_time = time.time()

        # Prepare tuning parameters
        tune_params = {
            "estimator": self.best_model_obj,
            "optimize": self.optimize_metric,
            "fold": self.config["tuning_folds"],
            "search_library": self.config["tuning_search_library"],
            "search_algorithm": self.config["tuning_search_algorithm"]
        }
        if self.config["tuning_search_algorithm"] == 'random':
             tune_params["n_iter"] = self.config["tuning_iterations"]
        # Add custom_grid if provided in config:
        # if "custom_grid" in self.config:
        #    tune_params["custom_grid"] = self.config["custom_grid"]

        try:
            self.tuned_best_model = self.pycaret_module.tune_model(**tune_params)
            logging.info(f"Model tuning completed in {time.time() - start_time:.2f} seconds.")
            logging.info("Tuned Model Parameters:")
            print(self.tuned_best_model) # Print pipeline for inspection
            return True
        except Exception as e:
            logging.error(f"Failed to tune model {best_model_id}: {e}", exc_info=True)
            return False

    def analyze_tuned_model(self) -> bool:
        """Evaluates and generates explanations for the tuned model."""
        if self.tuned_best_model is None or self.pycaret_module is None:
            logging.error("Tuned model or PyCaret module not available for analysis.")
            return False
        # --- Get Best Model ID ---
        best_model_id = "unknown_best" # Default fallback
        if self.results_df is not None and not self.results_df.empty:
            best_model_id = self.results_df.index[0]
        else:
             logging.warning("Results dataframe is empty or None when analyzing tuned model, cannot reliably get best model ID.")
             # Attempt to get model type directly from the pipeline if possible (more robust)
             try:
                 # Access the actual estimator step in the pipeline
                 # The name 'actual_estimator' is common in PyCaret >= 3.x, might vary in older versions
                 final_estimator = self.tuned_best_model.named_steps.get('actual_estimator', self.tuned_best_model.steps[-1][1])
                 # Get the model's class name (e.g., 'LogisticRegression')
                 model_class_name = final_estimator.__class__.__name__
                 # Map class name back to PyCaret ID (this mapping might need refinement)
                 # For now, we'll still rely on the ID from compare_models if available
                 logging.info(f"Actual estimator type in tuned pipeline: {model_class_name}")
             except Exception:
                 logging.warning("Could not determine actual estimator type from pipeline.")

        logging.info(f"--- Analyzing Tuned Model: {best_model_id} ---")
        try:
            logging.info("Evaluating Tuned Model...")
            _ = self.pycaret_module.evaluate_model(self.tuned_best_model)

            # Log feature importance plot (this usually works for most models)
            self._log_plot_artifact(
                 plot_function=lambda save, verbose: self.pycaret_module.plot_model(self.tuned_best_model, plot='feature', save=save, verbose=verbose),
                 plot_name=f"Feature Importance (Tuned {best_model_id})"
            )

            # --- Conditionally Log SHAP Summary Plot ---
            # Define models supported by interpret_model(plot='summary') based on error msg
            supported_shap_summary_models = {'et', 'lightgbm', 'xgboost', 'rf', 'dt', 'catboost'}

            # Check if the best model *type* (using the ID) is supported and if it's a classification task
            if self.task_type == 'classification' and best_model_id in supported_shap_summary_models:
                logging.info(f"Model type '{best_model_id}' supports SHAP summary plot. Generating...")
                self._log_plot_artifact(
                     plot_function=lambda save, verbose: self.pycaret_module.interpret_model(self.tuned_best_model, plot='summary', save=save), # interpret doesn't use verbose
                     plot_name=f"SHAP Summary Plot (Tuned {best_model_id})"
                )
            else:
                # Log a warning instead of trying to generate the unsupported plot
                logging.warning(f"SHAP summary plot via interpret_model() is not supported by PyCaret for model type '{best_model_id}' in task '{self.task_type}'. Skipping SHAP summary plot generation.")
                # Note: Other SHAP plots might still be possible for 'lr' using different methods or plot types if needed.
            return True
        except Exception as e:
            logging.error(f"Failed during tuned model analysis: {e}", exc_info=True)
            return False
        
        
    def finalize_and_save_model(self) -> bool:
        """Finalizes the tuned model and saves it."""
        if self.tuned_best_model is None or self.pycaret_module is None:
            logging.error("Tuned model or PyCaret module not available for finalization.")
            return False
        if self.results_df is None or self.results_df.empty:
             logging.warning("Results dataframe is empty, cannot get best model ID.")
             best_model_id = "unknown_best" # Fallback ID
        else:
            best_model_id = self.results_df.index[0]

        logging.info("--- Finalizing and Saving Model ---")
        try:
            self.final_model = self.pycaret_module.finalize_model(self.tuned_best_model)
            logging.info(f"Model finalized using pipeline: {self.final_model.__class__.__name__}")

            # Save final model artifact
            save_path = self._save_model_artifact(self.final_model, "final", best_model_id)
            return save_path is not None # Return True if saving was successful
        except Exception as e:
            logging.error(f"Failed to finalize or save model: {e}", exc_info=True)
            return False


    def run_pipeline(self):
        """Executes the full AutoML pipeline sequentially."""
        start_run_time = time.time()
        run_status = "FAILED" # Default status

        # Generate a unique run name including task type and timestamp
        run_name = f"AutoML_{self.config.get('experiment_name', 'exp')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:
             mlflow_run_id = run.info.run_id
             logging.info(f"Starting MLflow Run: {run_name} (ID: {mlflow_run_id})")
             mlflow.log_params(self.config) # Log all config parameters
             mlflow.set_tag("pipeline_status", "started")

             try:
                if not self.load_data(): return
                if not self.detect_task_type(): return
                # Log detected task type after detection
                mlflow.log_param("detected_task_type", self.task_type)

                if not self.setup_pycaret(): return
                if not self.compare_models(): return
                if not self.analyze_baseline_models():
                    logging.warning("Baseline model analysis did not complete successfully.")
                    # Decide if pipeline should continue despite baseline failures

                if not self.tune_best_model(): return
                if not self.analyze_tuned_model():
                    logging.warning("Tuned model analysis did not complete successfully.")
                    # Decide if pipeline should continue

                if not self.finalize_and_save_model(): return

                run_status = "COMPLETED"
                logging.info("--- AutoML Pipeline Completed Successfully ---")

             except Exception as e:
                 run_status = "ERROR"
                 logging.error(f"--- AutoML Pipeline Failed: {e} ---", exc_info=True)
                 mlflow.set_tag("error_message", str(e))

             finally:
                 end_run_time = time.time()
                 duration = end_run_time - start_run_time
                 mlflow.log_metric("pipeline_duration_seconds", duration)
                 mlflow.set_tag("pipeline_status", run_status)
                 logging.info(f"MLflow Run {mlflow_run_id} finished with status: {run_status}. Duration: {duration:.2f} seconds.")
                 # MLflow run context automatically ends the run


    def predict_on_new_data(self, new_data: pd.DataFrame, model_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Makes predictions on new data using a saved model pipeline.

        Args:
            new_data: DataFrame with the same features as the training data (excluding target).
            model_path: Optional path to a specific .pkl model file (without extension).
                        If None, loads the latest 'final' model from the configured save directory.

        Returns:
            DataFrame with predictions, or None if prediction fails.
        """
        logging.info("--- Making Predictions ---")

        # Determine which model to load
        load_path = model_path
        if load_path is None:
            try:
                final_models = [f for f in os.listdir(self.config["model_save_dir"]) if f.startswith('final_') and f.endswith('.pkl')]
                if not final_models:
                    logging.error(f"No final models found in '{self.config['model_save_dir']}'.")
                    return None
                latest_final_model_file = max(final_models, key=lambda f: os.path.getmtime(os.path.join(self.config["model_save_dir"], f)))
                load_path = os.path.join(self.config["model_save_dir"], latest_final_model_file.replace('.pkl',''))
                logging.info(f"Loading latest final model: {load_path}.pkl")
            except Exception as e:
                logging.error(f"Error finding latest model: {e}", exc_info=True)
                return None
        else:
             logging.info(f"Loading specified model: {load_path}.pkl")

        # Determine the correct PyCaret module (need task type)
        # This assumes the runner instance has already run detect_task_type or task_type is set
        if self.task_type == 'classification':
            pred_module = pyclf
        elif self.task_type == 'regression':
            pred_module = pyreg
        else:
             # If task_type isn't known, we might need to infer it from the loaded model/pipeline
             # Or require it to be passed explicitly for prediction.
             # For simplicity, let's try loading with clf and fallback to reg if needed,
             # but this isn't robust. Best practice would be to store task_type with the model.
             logging.warning("Task type not definitively known for prediction. Attempting classification module.")
             pred_module = pyclf
             # A better approach: Save task_type alongside the model or infer from pipeline object.

        try:
            loaded_pipeline = pred_module.load_model(load_path)
            logging.info(f"Model loaded successfully from {load_path}.pkl")

            predictions = pred_module.predict_model(loaded_pipeline, data=new_data)
            logging.info("Predictions generated successfully.")
            return predictions

        except ImportError: # Handle case where module might mismatch (e.g. tried loading reg model with clf)
            if pred_module == pyclf:
                logging.warning("Failed loading with classification module, trying regression...")
                try:
                    loaded_pipeline = pyreg.load_model(load_path)
                    predictions = pyreg.predict_model(loaded_pipeline, data=new_data)
                    logging.info("Predictions generated successfully using regression module.")
                    return predictions
                except Exception as e_reg:
                    logging.error(f"Failed to load model with both classification and regression modules: {e_reg}", exc_info=True)
                    return None
            else: # Should not happen with current logic but good practice
                 logging.error(f"Failed to load model with {pred_module.__name__}: Unknown error", exc_info=True)
                 return None
        except FileNotFoundError:
             logging.error(f"Could not load model file: {load_path}.pkl not found.")
             return None
        except Exception as e:
             logging.error(f"An error occurred during prediction: {e}", exc_info=True)
             return None

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting AutoML Runner Script")
    print(f"PyCaret Version: {pycaret.__version__}")
    print(f"MLflow Version: {mlflow.__version__}")


    # 1. Initialize the runner with configuration
    runner = AutoMLRunner(config=CONFIG)

    # 2. Run the main pipeline
    runner.run_pipeline()

    # 3. Example Prediction (Optional)
    # Create some dummy new data for prediction demonstration
    # IMPORTANT: In a real scenario, ensure the dummy data schema (columns and types)
    # exactly matches the data used for training (excluding the target).
    # Creating dummy data robustly requires knowing the original features and types.
    logging.info("\n--- Running Prediction Example ---")
    if runner.data is not None and runner.task_type is not None: # Check if data was loaded
        try:
            feature_columns = runner.data.drop(columns=[CONFIG["target_column"]]).columns
            dummy_data_dict = {}
            original_data_for_types = runner.data.drop(columns=[CONFIG["target_column"]])

            # Attempt to create dummy data based on training data stats
            # This is a simplified example and might fail for complex types or edge cases
            example_row = {}
            for col in feature_columns:
                 col_dtype = original_data_for_types[col].dtype
                 if pd.api.types.is_numeric_dtype(col_dtype):
                     example_row[col] = [original_data_for_types[col].mean()]
                 else:
                     mode_val = original_data_for_types[col].mode()
                     example_row[col] = [mode_val[0]] if not mode_val.empty else [None] # Use None if mode is empty

            new_data_example = pd.DataFrame(example_row)
            logging.info("\nExample new data row for prediction:")
            print(new_data_example)

            # Make prediction using the LATEST saved final model
            predictions_df = runner.predict_on_new_data(new_data=new_data_example)

            if predictions_df is not None:
                logging.info("\nPredictions on new data:")
                print(predictions_df)
            else:
                logging.warning("Prediction example failed.")

        except Exception as pred_ex:
            logging.error(f"Failed to create dummy data or run prediction example: {pred_ex}", exc_info=True)
    else:
        logging.warning("Skipping prediction example as initial data was not loaded or task type not detected.")

    logging.info("AutoML Runner Script Finished")