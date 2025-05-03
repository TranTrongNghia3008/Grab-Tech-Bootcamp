import pandas as pd
import pycaret.classification as pyclf
import pycaret.regression as pyreg
from sklearn.pipeline import Pipeline
import time
import numpy as np
import os
import joblib
import mlflow
import datetime
import logging
import json
import re
from typing import List, Optional, Dict, Any, Tuple, Union

# Import new dependencies
try:
    from ydata_profiling import ProfileReport
except ImportError:
    ProfileReport = None
    logging.warning("ydata-profiling not found. Run 'pip install ydata-profiling' to enable data profiling.")

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except ImportError:
    Report = None
    DataDriftPreset = None
    logging.warning("evidently not found. Run 'pip install evidently' to enable data drift detection.")

try:
    import shap
except ImportError:
    shap = None
    logging.warning("shap not found. Run 'pip install shap' to enable prediction explanations.")

import pycaret # Keep for version check if needed elsewhere


# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# (Paste Updated CONFIG dictionary here)
CONFIG = {
    # --- Core Paths & Identifiers ---
    "data_file_path": 'E:/Grab Bootcamp/Grab-Tech-Bootcamp/ML/automl/train.csv', # UPDATE THIS PATH
    "target_column": 'Survived',
    "experiment_name": 'automl_enhanced_exp_v1',
    "model_save_dir": './automl_models_enhanced',
    "mlflow_tracking_uri": "sqlite:///mlflow_enhanced.db",
    "session_id": 123,

    # --- Feature Selection ---
    "feature_columns": None, # Optional: List of feature names. If None, use all except target.

    # --- Setup Enhancements ---
    "run_data_profiling": True, # Generate profile report during setup?
    "profile_report_name": "data_profile_report.html",
    "schema_validation_path": None, # Optional: Path to JSON/YAML schema file
    # --- Advanced Preprocessing Options (Passed to pycaret.setup) ---
    "setup_params_extra": {
        # "feature_interaction": False, # Example: Set to True to enable
        # "polynomial_features": False, # Example: Set to True to enable
        # "bin_numeric_features": None, # Example: ['Age', 'Fare'] to bin these
        # "remove_outliers": False, # Example: Set to True to enable
        # "outlier_threshold": 0.05,
        # "feature_selection": False, # Example: Set to True to enable
        # "feature_selection_threshold": 0.8,
        # Add other pycaret setup args here as needed
        "numeric_imputation": 'mean', # Moved basic ones here too
        "categorical_imputation": 'mode',
        "normalize": False, # Default, adjust as needed (used below)
    },
    # --- Automated Feature Engineering (Basic - using PyCaret setup) ---
    "enable_feature_engineering": False, # Controls options like interaction/polynomial in setup_params_extra

    # --- Model Comparison Enhancements ---
    "baseline_classification_models": ['lr', 'dt'],
    "baseline_regression_models": ['lr', 'dt'],
    "sort_metric_classification": 'Accuracy',
    "sort_metric_regression": 'R2',
    "compare_include_models": None, # Optional: List of model IDs to include
    "compare_exclude_models": None, # Optional: List of model IDs to exclude
    "baseline_folds": 5,

    # --- Tuning Enhancements ---
    "optimize_metric_classification": 'Accuracy',
    "optimize_metric_regression": 'R2',
    "tuning_folds": 5,
    "tuning_search_library": 'scikit-learn', # 'scikit-learn', 'optuna', 'hyperopt' etc.
    "tuning_search_algorithm": 'random', # 'random' or 'grid'
    "tuning_iterations": 20, # Used for 'random' search or bayesian optimizers

    # --- Ensemble Modeling ---
    "ensemble_method": 'Bagging', # 'Bagging', 'Boosting'
    "ensemble_n_estimators": 10,

    # --- Analysis Enhancements ---
    "return_raw_explanations": True, # Attempt to return raw importance/SHAP values?

    # --- Finalize & Registry ---
    "generate_model_card": True,
    "register_model_in_mlflow": True, # Automatically register the final model?
    "mlflow_registered_model_name": "AutoML_Production_Candidate", # Name in MLflow Model Registry
    "mlflow_model_stage": "Staging", # Initial stage ('Staging', 'Production', 'Archived')

    # --- Data Drift ---
    "enable_drift_check_on_predict": True,
    "drift_report_name": "prediction_drift_report.html",

    # --- Prediction Explanation ---
    "enable_prediction_explanation": True
}


def sanitize_filename(name: str) -> str:
    """Removes or replaces characters unsuitable for filenames."""
    # Remove most special characters, replace spaces/dots with underscore
    name = re.sub(r'[^\w\-. ]', '', name) # Keep word chars, hyphen, dot, space
    name = re.sub(r'[ .]+', '_', name) # Replace spaces/dots with underscore
    return name

class AutoMLRunner:
    """
    Encapsulates an enhanced AutoML workflow using PyCaret and MLflow,
    allowing for granular execution and incorporating best practices.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.train_data: Optional[pd.DataFrame] = None # Data used by pycaret setup (post-split)
        self.test_data: Optional[pd.DataFrame] = None # Hold-out set from pycaret setup
        self.task_type: Optional[str] = None
        self.pycaret_module: Optional[Any] = None
        self.sort_metric: Optional[str] = None
        self.optimize_metric: Optional[str] = None
        self.baseline_model_ids: List[str] = []
        self.setup_env: Optional[Any] = None
        self.preprocessor: Optional[Any] = None # Fitted preprocessing pipeline
        self.best_model_obj: Optional[Any] = None
        self.results_df: Optional[pd.DataFrame] = None
        self.baseline_models_trained: Dict[str, Any] = {}
        self.tuned_model: Optional[Any] = None # Stores result of tune_model
        self.ensemble_model: Optional[Any] = None # Stores result of create_ensemble_model
        self.final_model: Optional[Any] = None
        self.active_mlflow_run_id: Optional[str] = None
        self.training_data_profile_path: Optional[str] = None # Path to saved data profile

        os.makedirs(self.config["model_save_dir"], exist_ok=True)
        self._setup_mlflow()

    # --- Internal Helper Methods ---



    
    
   # --- Internal Helper Methods ---
    def _generate_model_card(self, model_name, saved_model_base_path) -> str:
        """Generates a simple markdown model card string."""
        # --- Requires access to metrics, config etc stored on self ---
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        card = f"""
        # Model Card: {model_name}

        **Generated:** {timestamp}
        **MLflow Run ID:** {self.active_mlflow_run_id}

        ## Model Details
        - **Model Type:** (Attempt to infer from name/object) {model_name}
        - **Task Type:** {self.task_type}
        - **PyCaret Version:** {pycaret.__version__}
        - **Saved Path (Base):** `{saved_model_base_path}`
        - **MLflow Registered Name:** {self.config.get('mlflow_registered_model_name', 'N/A')}
        - **MLflow Stage:** {self.config.get('mlflow_model_stage', 'N/A')}

        ## Intended Use
        - *TODO: Describe the intended use case of this model.*

        ## Training Data
        - **Source Path:** `{self.config.get('data_file_path', 'N/A')}`
        - **Target Column:** `{self.config.get('target_column', 'N/A')}`
        - *Note: Refer to data profile artifact for details.*

        ## Performance Metrics
        - *TODO: Add key metrics from analysis step (e.g., hold-out accuracy/R2).*
        - *Example: Holdout Accuracy: {self.latest_analysis_metrics.get('Accuracy', 'N/A') if hasattr(self, 'latest_analysis_metrics') else 'N/A'}*

        ## Limitations & Bias
        - *TODO: Describe known limitations, potential biases, and ethical considerations.*

        ## Configuration Snapshot (Key Params)
        ```json
        {{
            "session_id": {self.config.get('session_id')},
            "optimize_metric": "{self.optimize_metric}",
            "tuning_library": "{self.config.get('tuning_search_library')}"
            # Add more key parameters here
        }}

        """
        # Note: Getting metrics reliably here requires storing them from the analyze step
        return card.strip()
    
    def _setup_mlflow(self):
        """Sets up MLflow tracking URI and experiment."""
        logging.info("--- Setting up MLflow Tracking ---")
        mlflow.set_tracking_uri(self.config["mlflow_tracking_uri"])
        mlflow.set_experiment(self.config["experiment_name"])
        logging.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        logging.info(f"MLflow experiment set to: {self.config['experiment_name']}")
        # Ensure no runs are active globally when initializing the runner instance
        while mlflow.active_run():
            logging.warning(f"Detected globally active run {mlflow.active_run().info.run_id} during runner initialization. Ending it.")
            mlflow.end_run(status="KILLED")

    def _start_mlflow_run(self, run_name_prefix="AutoML_Run"):
        """
        Starts a new MLflow run for this instance if one isn't already active,
        or returns the existing active run associated with this instance.
        Aggressively ends any unexpected "orphaned" or lingering active runs
        before attempting to start a new one.
        """
        instance_expected_run_id = self.active_mlflow_run_id # Store what instance expects

        # --- Aggressive Cleanup Loop ---
        # Try to ensure no run is active before we attempt to start our run.
        retry_count = 0
        max_retries = 5 # Prevent potential infinite loops
        while mlflow.active_run() is not None and retry_count < max_retries:
            active_run_obj = mlflow.active_run()
            active_run_id = active_run_obj.info.run_id
            logging.warning(f"[Cleanup Loop-{retry_count}] Detected active run {active_run_id} before starting intended run. Attempting to end it.")
            try:
                status_to_set = "KILLED" # Mark as killed if ended unexpectedly
                mlflow.end_run(status=status_to_set)
                logging.info(f"[Cleanup Loop-{retry_count}] Ended run {active_run_id} with status {status_to_set}.")
            except Exception as e:
                logging.error(f"[Cleanup Loop-{retry_count}] Failed to end active run {active_run_id}: {e}. Retrying check.")
            retry_count += 1
            time.sleep(0.1) # Give MLflow a very brief moment to update state

        # Check if cleanup succeeded
        if mlflow.active_run() is not None:
            final_active_run_id = mlflow.active_run().info.run_id
            logging.error(f"Failed to clear active MLflow run ({final_active_run_id}) after {max_retries} retries. Cannot start new run.")
            raise Exception(f"Failed to clear pre-existing active MLflow run {final_active_run_id}.")
        # --- End Aggressive Cleanup Loop ---

        # Now we should be certain no run is active.

        # If the instance expected a specific run, but it's gone now (due to cleanup or other reasons), reset instance state.
        if instance_expected_run_id:
            logging.warning(f"Runner instance expected active run {instance_expected_run_id}, but no run is active now (post-cleanup). Resetting instance state.")
            self.active_mlflow_run_id = None

        # Proceed to start the new run for this instance
        run_name = f"{run_name_prefix}_{self.task_type or 'unknown'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            run = mlflow.start_run(run_name=run_name)
            self.active_mlflow_run_id = run.info.run_id
            logging.info(f"Started new MLflow Run: {run_name} (ID: {self.active_mlflow_run_id})")
            # Log initial params/tags for the new run
            mlflow.log_params({k: v for k, v in self.config.items() if not isinstance(v, (dict, list))})
            mlflow.log_dict(self.config.get("setup_params_extra", {}), "pycaret_setup_params_extra.json")
            mlflow.set_tag("pipeline_stage", "started")
            return run
        except Exception as e:
            # This *shouldn't* happen now if the cleanup worked, but catch just in case.
            logging.error(f"Failed to start a new MLflow run even after cleanup: {e}", exc_info=True)
            self.active_mlflow_run_id = None # Ensure state is reset on failure
            raise # Re-raise the exception as starting the run is critical

    def _end_mlflow_run(self, status="FINISHED"):
        """
        Ends the active MLflow run associated with this instance, if one exists and is active.
        """
        current_active_run = mlflow.active_run()
        instance_run_id = self.active_mlflow_run_id # Store locally before clearing

        if instance_run_id:
            if current_active_run and current_active_run.info.run_id == instance_run_id:
                logging.info(f"Attempting to end MLflow run {instance_run_id} associated with this instance.")
                try:
                    current_status = current_active_run.info.status
                    final_status = status if current_status == "RUNNING" else current_status # Don't override a terminal status
                    mlflow.set_tag("pipeline_stage", "finished") # Set tag before ending
                    mlflow.end_run(status=final_status)
                    logging.info(f"MLflow Run {instance_run_id} ended with status: {final_status}")
                except Exception as e:
                    logging.error(f"Error ending MLflow run {instance_run_id}: {e}", exc_info=True)
                finally:
                    # Always clear the instance's run ID after attempting to end
                    if self.active_mlflow_run_id == instance_run_id:
                         self.active_mlflow_run_id = None
            else:
                # Instance thought a run was active, but it wasn't found or didn't match
                logging.warning(f"Attempted to end run {instance_run_id}, but it was not the globally active run (Active: {current_active_run.info.run_id if current_active_run else 'None'}). Clearing instance run ID.")
                self.active_mlflow_run_id = None
        else:
            logging.debug("No active run associated with this runner instance to end.")
   
    def _log_plot_artifact(self, plot_function: callable, base_plot_name: str, model_name: str, **kwargs):
        """
        Helper to generate, save (with unique name), and log a plot artifact.

        Args:
            plot_function: The callable that generates the plot (e.g., plot_model).
                        Must accept save=True.
            base_plot_name: The generic name for the plot type (e.g., "Feature Importance").
            model_name: The specific name of the model for uniqueness (e.g., "baseline_lr").
            **kwargs: Additional arguments for the plot_function.
        """
        if not self.active_mlflow_run_id:
            logging.warning("No active MLflow run to log plot artifact.")
            return
        # No need to call _start_mlflow_run here if called from within methods already in run context
        # self._start_mlflow_run()

        logging.info(f"Generating '{base_plot_name}' plot for model '{model_name}'...")
        original_save_path = None
        new_save_path = None

        try:
            # Let pycaret save with its default name first
            original_save_path = plot_function(save=True, verbose=False, **kwargs)
            logging.info(f"Plot saved temporarily to: {original_save_path}")

            if original_save_path and os.path.exists(original_save_path):
                try:
                    # --- Construct new unique filename ---
                    dir_name = os.path.dirname(original_save_path)
                    original_filename_with_ext = os.path.basename(original_save_path)
                    # Use the base_plot_name provided for consistency, get extension from original
                    _ , file_ext = os.path.splitext(original_filename_with_ext)
                    if not file_ext: file_ext = ".png" # Default extension if needed

                    sanitized_model_name = sanitize_filename(model_name)
                    sanitized_base_plot_name = sanitize_filename(base_plot_name)
                    new_filename = f"{sanitized_base_plot_name}_{sanitized_model_name}{file_ext}"
                    new_save_path = os.path.join(dir_name, new_filename)

                    # --- Rename the file ---
                    logging.debug(f"Attempting to rename '{original_save_path}' to '{new_save_path}'")
                    os.rename(original_save_path, new_save_path)
                    logging.info(f"Renamed plot file to: {new_save_path}")

                    # --- Log the renamed file ---
                    mlflow.log_artifact(new_save_path)
                    logging.info(f"Successfully logged plot artifact: {new_save_path}")

                except Exception as log_rename_e:
                    logging.error(f"Could not rename or log plot artifact. Original path: {original_save_path}, Target path: {new_save_path}. Error: {log_rename_e}", exc_info=True)
                    # Optionally try logging the original path as a fallback
                    if original_save_path and os.path.exists(original_save_path):
                        try:
                            mlflow.log_artifact(original_save_path, artifact_path="plots_unrenamed") # Log to subfolder
                            logging.warning(f"Logged plot artifact with original name to 'plots_unrenamed': {original_save_path}")
                        except Exception as fallback_e:
                            logging.error(f"Fallback logging failed for {original_save_path}: {fallback_e}")

            else:
                logging.warning(f"Plot file not found or path invalid after generation for '{base_plot_name}' model '{model_name}'. Expected path: {original_save_path}")

        except Exception as plot_e:
            logging.error(f"Failed to generate or save plot '{base_plot_name}' for model '{model_name}': {plot_e}", exc_info=True)

    def _save_model_artifact(self, model_object: Any, model_stage_prefix: str, model_id: str) -> Optional[str]:
        """Helper to save a model artifact with timestamp and log path."""
        if self.pycaret_module is None: return None
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename_base = f"{model_stage_prefix}_{model_id}_{timestamp}"
            save_path_base = os.path.join(self.config["model_save_dir"], model_filename_base)

            self.pycaret_module.save_model(model_object, save_path_base)
            full_save_path_pkl = f"{save_path_base}.pkl"
            full_save_path_meta = f"{save_path_base}_meta.json"

            logging.info(f"Model '{model_id}' ({model_stage_prefix}) saved to {full_save_path_pkl}")

            # --- Save Task Type Metadata ---
            metadata = {"task_type": self.task_type, "pycaret_version": pycaret.__version__}
            with open(full_save_path_meta, 'w') as f:
                json.dump(metadata, f)
            logging.info(f"Metadata saved to {full_save_path_meta}")

            if self.active_mlflow_run_id:
                 self._start_mlflow_run()
                 mlflow.log_param(f"{model_stage_prefix}_{model_id}_save_path_pkl", full_save_path_pkl)
                 mlflow.log_param(f"{model_stage_prefix}_{model_id}_save_path_meta", full_save_path_meta)
                 # Log metadata file as artifact
                 mlflow.log_artifact(full_save_path_meta)
                 # Optionally log model itself - PyCaret often does this if log_experiment=True in setup
                 # mlflow.pycaret.log_model(model_object, f"{model_stage_prefix}_{model_id}")

            return save_path_base # Return base path without extension
        except Exception as e:
            logging.error(f"Failed to save model artifact {model_stage_prefix} {model_id}: {e}", exc_info=True)
            return None

    def _load_model_and_metadata(self, model_base_path: str) -> Tuple[Optional[Any], Optional[str]]:
        """Loads model .pkl and associated _meta.json, returns model object and task type."""
        model_pkl_path = f"{model_base_path}.pkl"
        model_meta_path = f"{model_base_path}_meta.json"
        loaded_model = None
        task_type = None

        if not os.path.exists(model_pkl_path):
            logging.error(f"Model file not found: {model_pkl_path}")
            return None, None

        if os.path.exists(model_meta_path):
            try:
                with open(model_meta_path, 'r') as f:
                    metadata = json.load(f)
                task_type = metadata.get("task_type")
                logging.info(f"Loaded task type '{task_type}' from {model_meta_path}")
            except Exception as e:
                logging.warning(f"Could not load or parse metadata file {model_meta_path}: {e}. Task type remains unknown.")
        else:
             logging.warning(f"Metadata file {model_meta_path} not found. Task type remains unknown.")


        # Determine module based on loaded task_type if available, otherwise fallback
        if task_type == 'classification':
            load_module = pyclf
        elif task_type == 'regression':
            load_module = pyreg
        else:
            logging.warning("Task type unknown/invalid from metadata, attempting load with classification module.")
            load_module = pyclf # Default guess

        try:
            loaded_model = load_module.load_model(model_base_path, verbose=False)
            logging.info(f"Model loaded successfully from {model_base_path}.pkl")
            return loaded_model, task_type
        except Exception as e_load:
            logging.error(f"Failed loading model {model_base_path}.pkl with {load_module.__name__}: {e_load}")
            # Try fallback if first guess failed
            if load_module == pyclf and task_type is None:
                 logging.info("Trying fallback load with regression module...")
                 try:
                     loaded_model = pyreg.load_model(model_base_path, verbose=False)
                     logging.info(f"Model loaded successfully from {model_base_path}.pkl using regression module.")
                     return loaded_model, 'regression' # Assume regression if this worked
                 except Exception as e_fallback:
                     logging.error(f"Fallback load failed: {e_fallback}")
                     return None, None
            return None, None

    # --- Feature Functions ---
    # set up necessary environment for AutoML and Pycaret for AUTOML PIPELINE
    def setup_automl_environment(self, data_path: str, target_column: str, feature_columns: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
        """
        Loads data, validates, detects task type, profiles data (optional),
        and initializes the PyCaret environment. Mandatory first step.
        """
        logging.info("--- 1. Setting Up Environment ---")
        self.config["data_file_path"] = data_path # Update config with actual paths used
        self.config["target_column"] = target_column
        self.config["feature_columns"] = feature_columns

        # 1.1 Load Data
        logging.info("--- 1.1 Loading Data ---")
        try:
            self.data = pd.read_csv(data_path)
            logging.info(f"Data loaded successfully with shape: {self.data.shape}")
            if target_column not in self.data.columns:
                raise ValueError(f"Target column '{target_column}' not found.")
            if feature_columns: # Validate provided feature columns
                 missing_features = [col for col in feature_columns if col not in self.data.columns]
                 if missing_features:
                     raise ValueError(f"Specified feature columns not found in data: {missing_features}")
            self.data = self.data.reset_index(drop=True)
        except Exception as e:
            logging.error(f"Error loading data: {e}", exc_info=True)
            return False, None

        # 1.2 Schema Validation (Optional)
        if self.config.get("schema_validation_path"):
            logging.warning("Schema validation requested but not implemented in this version.")
            # Placeholder: Implement schema loading and validation logic here

        # 1.3 Data Profiling (Optional)
        if self.config.get("run_data_profiling") and ProfileReport:
            logging.info("--- 1.2 Generating Data Profile Report ---")
            try:
                profile = ProfileReport(self.data, title="Data Profiling Report", minimal=True) # Use minimal=True for speed
                report_path = self.config.get("profile_report_name", "data_profile_report.html")
                profile.to_file(report_path)
                logging.info(f"Data profile report saved to {report_path}")
                self.training_data_profile_path = report_path # Store for potential drift check
                # Log profile to MLflow (optional, can be large)
                # self._start_mlflow_run("Data_Profiling") # May start run too early
                # mlflow.log_artifact(report_path)
                # self._end_mlflow_run()
            except Exception as e:
                logging.error(f"Failed to generate data profile: {e}", exc_info=True)
                # Continue even if profiling fails

        # 1.4 Detect Task Type
        logging.info("--- 1.3 Auto-Detecting Task Type ---")
        # (Using the same logic as before, simplified here)
        target_series = self.data[target_column]
        n_unique = target_series.nunique()
        dtype = target_series.dtype
        if pd.api.types.is_numeric_dtype(dtype) and n_unique >= self.config.get("unique_value_threshold_for_classification", 20) and not (len(target_series.unique()) == 2 and target_series.isin([0, 1]).all()):
             self.task_type = 'regression'
        else:
             self.task_type = 'classification' # Default or explicit classification cases
        logging.info(f"Detected task type: {self.task_type}")


        # 1.5 Setup PyCaret Environment
        logging.info("--- 1.4 Setting up PyCaret Environment ---")
        start_time = time.time()

        setup_params = {
            "data": self.data,
            "target": target_column,
            "session_id": self.config["session_id"],
            "log_experiment": True, # Let PyCaret log to our active run
            "experiment_name": self.config["experiment_name"],
            "verbose": False, # Reduce PyCaret's console noise
            # Add other setup parameters dynamically
            **self.config.get("setup_params_extra", {})
        }

        # Handle feature selection based on input
        if feature_columns:
            setup_params["use_features"] = feature_columns
            logging.info(f"Using specified features: {feature_columns}")
        # Note: 'ignore_features' could also be used if that's preferred logic

        # Assign PyCaret module and metrics based on task type
        if self.task_type == 'classification':
            self.pycaret_module = pyclf
            self.sort_metric = self.config["sort_metric_classification"]
            self.optimize_metric = self.config["optimize_metric_classification"]
            self.baseline_model_ids = self.config["baseline_classification_models"]
        elif self.task_type == 'regression':
            self.pycaret_module = pyreg
            self.sort_metric = self.config["sort_metric_regression"]
            self.optimize_metric = self.config["optimize_metric_regression"]
            self.baseline_model_ids = self.config["baseline_regression_models"]
            # Apply regression-specific setup defaults if not overridden in setup_params_extra
            setup_params.setdefault("normalize", True) # Example default
        else:
            logging.error(f"Invalid task type determined: {self.task_type}")
            return False, self.task_type

        # Enable automated feature engineering options if configured
        if self.config.get("enable_feature_engineering"):
            setup_params["feature_interaction"] = self.config["setup_params_extra"].get("feature_interaction", True) # Default True if main flag is True
            setup_params["polynomial_features"] = self.config["setup_params_extra"].get("polynomial_features", True) # Default True if main flag is True
            logging.info("Automated feature engineering options enabled in setup.")


        try:
            # Setup expects NO active MLflow run, it manages its own logging start/end within setup
            # So we ensure no run is active here IF PyCaret manages its own logging.
            # However, our structure wants one run for the whole process.
            # Let's try starting the run *before* setup and see if pycaret logs to it.
            self._start_mlflow_run(f"Setup_{self.task_type}")
            mlflow.log_param("target_column", target_column)
            mlflow.log_param("feature_columns_used", json.dumps(feature_columns) if feature_columns else "All (except target)")
            mlflow.log_param("detected_task_type", self.task_type)
            # Log the profile report artifact if created
            if self.training_data_profile_path and os.path.exists(self.training_data_profile_path):
                 mlflow.log_artifact(self.training_data_profile_path)


            self.setup_env = self.pycaret_module.setup(**setup_params)
            self.preprocessor = self.setup_env.pipeline # Get the fitted preprocessor
            # Store train/test data if needed later (e.g. for drift baseline)
            self.train_data = self.setup_env.X_train.join(self.setup_env.y_train)
            self.test_data = self.setup_env.X_test.join(self.setup_env.y_test)

            logging.info(f"PyCaret setup completed in {time.time() - start_time:.2f} seconds.")
            mlflow.log_param("pycaret_setup_status", "success")
            mlflow.set_tag("pipeline_stage", "setup_complete")
            return True, self.task_type

        except Exception as e:
            logging.error(f"PyCaret setup failed: {e}", exc_info=True)
            if self.active_mlflow_run_id:
                 mlflow.log_param("pycaret_setup_status", "failed")
                 mlflow.set_tag("error_message", f"Setup Failed: {e}")
                 self._end_mlflow_run(status="FAILED")
            return False, self.task_type
    # GENERATE SURVEY LIST OF MODEL AND BASELINE ALONG METRICS
    def compare_models_and_baselines(self) -> Optional[Dict[str, Any]]:
        """
        Compares standard models, analyzes configured baseline models,
        and logs results. Assumes setup is complete.
        """
        if self.pycaret_module is None or self.sort_metric is None or self.setup_env is None:
            logging.error("PyCaret environment not set up. Run setup_automl_environment first.")
            return None

        logging.info("--- 2. Comparing Models & Analyzing Baselines ---")
        self._start_mlflow_run(f"Compare_{self.task_type}") # Ensure run active, get ID
        mlflow.set_tag("pipeline_stage", "comparing_models")

        # 2.1 Compare Models
        logging.info("--- 2.1 Comparing Models ---")
        start_time = time.time()
        compare_success = False
        try:
            compare_args = {
                "sort": self.sort_metric,
                "include": self.config.get("compare_include_models"),
                "exclude": self.config.get("compare_exclude_models"),
                "verbose": False
            }
            # Filter out None values
            compare_args = {k: v for k, v in compare_args.items() if v is not None}

            self.best_model_obj = self.pycaret_module.compare_models(**compare_args)
            self.results_df = self.pycaret_module.pull()
            logging.info("\nModel Comparison Results (Cross-Validated):")
            print(self.results_df.head())
            logging.info(f"Model comparison completed in {time.time() - start_time:.2f} seconds.")

            if self.results_df is not None:
                results_html = self.results_df.to_html(escape=False)
                mlflow.log_text(results_html, "compare_models_results.html")
                # Log best model score
                best_score = self.results_df.iloc[0][self.sort_metric]
                mlflow.log_metric(f"best_model_cv_{self.sort_metric}", best_score)

            compare_success = True
            mlflow.log_param("compare_models_status", "success")

        except Exception as e:
            logging.error(f"Model comparison failed: {e}", exc_info=True)
            mlflow.log_param("compare_models_status", "failed")
            mlflow.set_tag("error_message", f"Compare Failed: {e}")
            # Decide whether to proceed to baselines if compare fails


        # 2.2 Analyze Baseline Models
        logging.info("--- 2.2 Baseline Model Analysis ---")
        mlflow.set_tag("pipeline_stage", "analyzing_baselines")
        baseline_analysis_success = True
        baseline_metrics_summary = {}

        if not self.baseline_model_ids:
             logging.warning("No baseline models specified in config.")
        else:
            for baseline_id in  self.baseline_model_ids:
                logging.info(f"\nAnalyzing baseline model: {baseline_id}")
                nested_run = None # Initialize for safety in except block
                try:
                    # Start a nested run for each baseline for isolated logging
                    with mlflow.start_run(run_id=self.active_mlflow_run_id, nested=True) as nested_run:
                        mlflow.set_tag("model_type", "baseline")
                        mlflow.set_tag("model_id", baseline_id)

                        baseline_model = self.pycaret_module.create_model(
                            baseline_id, fold=self.config["baseline_folds"], verbose=False
                        )
                        self.baseline_models_trained[baseline_id] = baseline_model
                        logging.info(f"Baseline model {baseline_id} created.")

                        logging.info(f"Evaluating Baseline Model: {baseline_id} on hold-out set...")
                        # predict_model implicitly uses hold-out set after create_model
                        hold_out_predictions = self.pycaret_module.predict_model(baseline_model, verbose=False)
                        baseline_metrics_df = self.pycaret_module.pull() # metrics from predict_model evaluation

                        print(f"Baseline {baseline_id} Hold-out Metrics:\n{baseline_metrics_df}")

                        # --- FIX 1: Filter metrics before logging ---
                        metrics_to_log = {}
                        if baseline_metrics_df is not None and not baseline_metrics_df.empty:
                            raw_metrics_dict = baseline_metrics_df.iloc[0].to_dict()
                            baseline_metrics_summary[baseline_id] = raw_metrics_dict # Store raw dict with model name

                            for key, value in raw_metrics_dict.items():
                                # Explicitly skip non-numeric columns like 'Model'
                                if isinstance(value, (str, type(None))): # Skip strings and None
                                    if key.lower() != 'model': # Log if skipping unexpected non-numeric
                                        logging.debug(f"Skipping non-numeric/None metric '{key}' for baseline {baseline_id}")
                                    continue
                                try:
                                    # Attempt conversion, allows bools/ints to become floats
                                    metrics_to_log[key] = float(value)
                                except (ValueError, TypeError) as convert_err:
                                    logging.warning(f"Could not convert metric '{key}' ({value}) to float for {baseline_id}: {convert_err}")

                            logging.info(f"Logging numeric metrics for {baseline_id}: {metrics_to_log}")
                            if metrics_to_log:
                                mlflow.log_metrics(metrics_to_log)
                            else:
                                logging.warning(f"No valid numeric metrics found to log for baseline {baseline_id}.")
                        else:
                            logging.warning(f"Could not pull metrics for baseline {baseline_id} after prediction.")
                        # --- End FIX 1 ---

                        # Log feature importance plot
                        self._log_plot_artifact(
                            plot_function=lambda save, verbose, **kwargs: self.pycaret_module.plot_model(baseline_model, plot='feature', save=save, verbose=verbose, **kwargs),
                            base_plot_name=f"Feature Importance" , # Generic plot type name 
                            model_name = f"baseline_{baseline_id}"# Specific model identifier
                        )

                        # Save baseline model artifact locally
                        self._save_model_artifact(baseline_model, "baseline", baseline_id)
                        mlflow.set_tag("status", "completed")

                except Exception as e:
                    logging.error(f"Error processing baseline model '{baseline_id}': {e}", exc_info=False) # Keep log concise
                    baseline_analysis_success = False
                    if nested_run: # Check if nested run context was successfully entered
                         mlflow.set_tag("status", "failed")
                         # --- FIX 2: Use unique key for error parameter ---
                         error_param_key = f"error_{baseline_id}"
                         error_message = str(e)[:MLFLOW_PARAM_VALUE_MAX_LEN] # Truncate error message
                         mlflow.log_param(error_param_key, error_message)
                         logging.info(f"Logged error for baseline {baseline_id} to MLflow param '{error_param_key}'.")
                         # --- End FIX 2 ---
                    else:
                         logging.error("Failed to log error to MLflow as nested run was not active.")
                    continue # Continue to next baseline

        mlflow.log_param("baseline_analysis_status", "success" if baseline_analysis_success else "partial_failure")

        if not compare_success and not baseline_analysis_success:
             logging.error("Neither model comparison nor baseline analysis succeeded.")
             self._end_mlflow_run(status="FAILED")
             return None

        best_model_id_from_compare = self.results_df.index[0] if (self.results_df is not None and not self.results_df.empty) else None

        return {
            "best_model_id": best_model_id_from_compare,
            "comparison_results": self.results_df.to_dict() if self.results_df is not None else None,
            "baseline_metrics": baseline_metrics_summary
        }
    # 
    def tune_model(self, model_id: Optional[str] = None, use_best_from_compare: bool = False, custom_grid: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Tunes hyperparameters for the best model from comparison OR a specified model_id.
        """
        if self.pycaret_module is None or self.optimize_metric is None or self.setup_env is None:
            logging.error("PyCaret environment not set up. Run setup_automl_environment first.")
            return None

        self._start_mlflow_run() # Ensure run active
        mlflow.set_tag("pipeline_stage", "tuning_model")

        model_to_tune_obj = None
        model_to_tune_id = None

        if use_best_from_compare:
            if self.best_model_obj is None:
                logging.error("Cannot use best model from compare - compare_models_and_baselines must be run first.")
                return None
            model_to_tune_obj = self.best_model_obj
            model_to_tune_id = self.results_df.index[0] if (self.results_df is not None and not self.results_df.empty) else "best_unknown"
            logging.info(f"Selected model to tune: Best from comparison ({model_to_tune_id})")
        elif model_id:
            logging.info(f"Selected model to tune: Specified ID ({model_id})")
            model_to_tune_id = model_id
            # Need to create the model first before tuning if not using the one from compare_models
            try:
                model_to_tune_obj = self.pycaret_module.create_model(model_to_tune_id, verbose=False)
            except Exception as e:
                 logging.error(f"Failed to create model '{model_id}' for tuning: {e}", exc_info=True)
                 return None
        else:
            logging.error("Must either specify a model_id or set use_best_from_compare=True.")
            return None

        logging.info(f"--- 3. Tuning Model: {model_to_tune_id} ---")
        start_time = time.time()

        tune_params = {
            "estimator": model_to_tune_obj,
            "optimize": self.optimize_metric,
            "fold": self.config["tuning_folds"],
            "search_library": self.config["tuning_search_library"],
            "search_algorithm": self.config["tuning_search_algorithm"],
            "verbose": False
        }
        if self.config["tuning_search_algorithm"] == 'random':
             tune_params["n_iter"] = self.config["tuning_iterations"]
        if custom_grid:
            tune_params["custom_grid"] = custom_grid

        try:
            with mlflow.start_run(run_id=self.active_mlflow_run_id, nested=True) as nested_run:
                 mlflow.set_tag("model_type", "tuned")
                 mlflow.set_tag("model_id", model_to_tune_id)
                 mlflow.log_param("tuning_search_library", tune_params["search_library"])
                 mlflow.log_param("tuning_search_algorithm", tune_params["search_algorithm"])
                 if "n_iter" in tune_params: mlflow.log_param("tuning_n_iter", tune_params["n_iter"])
                 if custom_grid: mlflow.log_dict(custom_grid, "tuning_custom_grid.json")


                 self.tuned_model = self.pycaret_module.tune_model(**tune_params)
                 tuned_metrics_df = self.pycaret_module.pull() # Get CV results from tuning

                 logging.info(f"Model tuning completed in {time.time() - start_time:.2f} seconds.")
                 logging.info("Tuned Model Parameters:")
                 # Extract and log best parameters
                 best_params = self.tuned_model.get_params()
                 # Filter for actual estimator params if pipeline
                 estimator_step_name = 'actual_estimator' # Common in PyCaret 3+
                 final_estimator_params = {}
                 if hasattr(self.tuned_model, 'named_steps') and estimator_step_name in self.tuned_model.named_steps:
                      actual_estimator = self.tuned_model.named_steps[estimator_step_name]
                      final_estimator_params = actual_estimator.get_params()
                      logging.info(f"Best Tuned Params (Estimator): {final_estimator_params}")
                      mlflow.log_params({f"tuned_{k}": v for k,v in final_estimator_params.items() if isinstance(v, (str, int, float, bool))}) # Log simple params
                 else:
                     logging.warning("Could not extract final estimator parameters easily.")
                     print(best_params)


                 logging.info(f"Tuned Model CV Metrics:\n{tuned_metrics_df}")
                 # Log mean CV metrics from tuning
                 mean_metrics = tuned_metrics_df.loc['Mean'].to_dict()
                 mlflow.log_metrics({f"tuned_cv_{k}": v for k, v in mean_metrics.items()})
                 mlflow.set_tag("status", "completed")


                 return {
                    "tuned_model_object": self.tuned_model, # Returning object for chaining
                    "tuned_model_id": model_to_tune_id,
                    "best_params": final_estimator_params or best_params,
                    "cv_metrics": mean_metrics
                }

        except Exception as e:
            logging.error(f"Failed to tune model {model_to_tune_id}: {e}", exc_info=True)
            if 'nested_run' in locals() and nested_run:
                 mlflow.set_tag("status", "failed")
                 mlflow.log_param("error", str(e))
            return None

    def create_ensemble_model(self, base_model_object: Any, base_model_id: str) -> Optional[Dict[str, Any]]:
        """Creates an ensemble model using the provided base model."""
        if self.pycaret_module is None or self.setup_env is None:
            logging.error("PyCaret environment not set up.")
            return None
        if base_model_object is None:
             logging.error("Base model object must be provided.")
             return None

        self._start_mlflow_run()
        mlflow.set_tag("pipeline_stage", "ensembling")

        method = self.config["ensemble_method"]
        n_estimators = self.config["ensemble_n_estimators"]
        ensemble_model_name = f"{method.lower()}_{base_model_id}"

        logging.info(f"--- 4. Creating Ensemble Model ---")
        logging.info(f"Method: {method}, Base Model: {base_model_id}, N Estimators: {n_estimators}")
        start_time = time.time()

        try:
            with mlflow.start_run(run_id=self.active_mlflow_run_id, nested=True) as nested_run:
                mlflow.set_tag("model_type", "ensemble")
                mlflow.set_tag("model_id", ensemble_model_name)
                mlflow.log_param("ensemble_method", method)
                mlflow.log_param("ensemble_n_estimators", n_estimators)
                mlflow.log_param("base_model_id", base_model_id)

                self.ensemble_model = self.pycaret_module.ensemble_model(
                    base_model_object,
                    method=method,
                    n_estimators=n_estimators,
                    fold=self.config.get("tuning_folds", 5), # Use tuning folds for consistency
                    optimize=self.optimize_metric, # Optimize metric for ensembling CV score
                    verbose=False
                )
                ensemble_metrics_df = self.pycaret_module.pull() # Get CV results

                logging.info(f"Ensemble model creation completed in {time.time() - start_time:.2f} seconds.")
                logging.info(f"Ensemble Model CV Metrics:\n{ensemble_metrics_df}")

                mean_metrics = ensemble_metrics_df.loc['Mean'].to_dict()
                mlflow.log_metrics({f"ensemble_cv_{k}": v for k, v in mean_metrics.items()})
                mlflow.set_tag("status", "completed")

                return {
                     "ensemble_model_object": self.ensemble_model,
                     "ensemble_model_id": ensemble_model_name,
                     "cv_metrics": mean_metrics
                 }
        except Exception as e:
            logging.error(f"Failed to create ensemble model: {e}", exc_info=True)
            if 'nested_run' in locals() and nested_run:
                 mlflow.set_tag("status", "failed")
                 mlflow.log_param("error", str(e))
            return None

    def analyze_trained_model(self, model_object: Any, model_name: str) -> Optional[Dict[str, Any]]:
        """Evaluates a trained model on hold-out and logs explanations."""
        if self.pycaret_module is None or self.setup_env is None:
            logging.error("PyCaret environment not set up.")
            return None
        if model_object is None:
             logging.error("Model object must be provided for analysis.")
             return None

        self._start_mlflow_run()
        mlflow.set_tag("pipeline_stage", f"analyzing_{model_name}")
        logging.info(f"--- 5. Analyzing Trained Model: {model_name} ---")

        analysis_results = {}
        feature_importance_data = None
        shap_values_data = None

        try:
             # Use a nested run for analysis steps of this specific model
            with mlflow.start_run(run_id=self.active_mlflow_run_id, nested=True) as nested_run:
                mlflow.set_tag("analysis_target_model", model_name)
                logging.info("Evaluating model on hold-out set...")
                # Use predict_model on the hold-out set (implicitly available after setup)
                hold_out_predictions = self.pycaret_module.predict_model(model_object, verbose=False)
                # Pull the metrics DataFrame generated by predict_model's evaluation
                hold_out_metrics_df = self.pycaret_module.pull()

                hold_out_metrics = {} # Initialize empty dict
                if hold_out_metrics_df is not None and not hold_out_metrics_df.empty:
                    hold_out_metrics = hold_out_metrics_df.iloc[0].to_dict()
                    logging.info(f"Hold-out Metrics for {model_name}:\n{pd.Series(hold_out_metrics).to_string()}")
                    analysis_results["holdout_metrics"] = hold_out_metrics # Store original full dict

                    # --- FIX: Filter metrics before logging ---
                    metrics_to_log = {}
                    for key, value in hold_out_metrics.items():
                        # Explicitly skip non-numeric columns like 'Model'
                        if isinstance(value, (str, type(None))):
                            if key.lower() != 'model': # Log if skipping unexpected non-numeric
                                logging.debug(f"Skipping non-numeric/None holdout metric '{key}' for {model_name}")
                            continue
                        try:
                            # Add prefix and attempt conversion
                            metrics_to_log[f"holdout_{key}"] = float(value)
                        except (ValueError, TypeError) as convert_err:
                             logging.warning(f"Could not convert holdout metric '{key}' ({value}) to float for {model_name}: {convert_err}")

                    if metrics_to_log:
                         logging.info(f"Logging numeric holdout metrics for {model_name}: {metrics_to_log}")
                         mlflow.log_metrics(metrics_to_log) # Log the filtered dict
                    else:
                         logging.warning(f"No valid numeric holdout metrics found to log for {model_name}.")
                    # --- End FIX ---
                else:
                    logging.warning(f"Could not pull hold-out metrics for {model_name}.")
                    analysis_results["holdout_metrics"] = None

                # --- FIX: Check for feature importance support before plotting ---
                logging.info("Checking for Feature Importance support...")
                # Need to check the actual final estimator, especially if model_object is a pipeline
                estimator_to_check = model_object
                is_pipeline = False
                if hasattr(model_object, 'steps'):
                    is_pipeline = True
                    # Try common PyCaret step name first, then fallback to last step
                    if hasattr(model_object, 'named_steps') and 'actual_estimator' in model_object.named_steps:
                        estimator_to_check = model_object.named_steps['actual_estimator']
                    else:
                        estimator_to_check = model_object.steps[-1][1] # Get estimator from last step tuple

                has_importance_attr = hasattr(estimator_to_check, 'feature_importances_') or \
                                      hasattr(estimator_to_check, 'coef_')

                analysis_results["feature_importances"] = None # Default unless extracted
                analysis_results["feature_importances_plot_generated"] = False

                if has_importance_attr:
                    logging.info(f"Model type '{type(estimator_to_check).__name__}' supports feature importance/coefficients.")
                    # Try extracting raw importance if requested and possible
                    if self.config.get("return_raw_explanations"):
                         try:
                             if hasattr(estimator_to_check, 'feature_importances_'):
                                 importances = estimator_to_check.feature_importances_
                                 importance_attr_name = 'feature_importances_'
                             elif hasattr(estimator_to_check, 'coef_'):
                                 # For coef_, often need absolute value; might be multi-dim for multi-class
                                 if estimator_to_check.coef_.ndim == 1: # Binary clf or regression
                                    importances = np.abs(estimator_to_check.coef_)
                                 else: # Multi-class, average abs coef across classes
                                    importances = np.mean(np.abs(estimator_to_check.coef_), axis=0)
                                 importance_attr_name = 'coef_'
                             else:
                                  raise AttributeError("Attribute found by hasattr but not accessible?")

                             # Get feature names AFTER preprocessing
                             feature_names = self.setup_env.X_train.columns # Correct place to get post-transform names
                             if len(importances) == len(feature_names):
                                  importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
                                  feature_importance_data = importance_df.to_dict(orient='records')
                                  analysis_results["feature_importances"] = feature_importance_data
                                  logging.info(f"Extracted raw feature importance/coefficients based on '{importance_attr_name}'.")
                             else:
                                 logging.warning(f"Mismatch between number of importances ({len(importances)}) and feature names ({len(feature_names)}). Skipping raw data extraction.")

                         except Exception as raw_e:
                              logging.error(f"Could not extract raw feature importances: {raw_e}")

                    # Generate and log the plot artifact
                    self._log_plot_artifact(
                         plot_function=lambda save, verbose, **kwargs: self.pycaret_module.plot_model(model_object, plot='feature', save=save, verbose=verbose, **kwargs),
                         base_plot_name="Feature Importance", # Generic plot type name
                        model_name= model_name # Specific model identifier
                    )
                    analysis_results["feature_importances_plot_generated"] = True

                else:
                    # Log a warning if the plot cannot be generated
                    logging.warning(f"Feature importance plot ('coef_' or 'feature_importances_') is not directly available via plot_model for the final estimator type '{type(estimator_to_check).__name__}' in model '{model_name}'. Skipping plot generation.")
                # --- End FIX ---


                # --- Log feature importance plot ---
                # (Keep existing feature importance logging logic here)
                logging.info("Generating Feature Importance...")
                try:
                     # Try to get raw importance values first
                     if hasattr(model_object, 'feature_importances_'): # Simple case for single models
                          importances = model_object.feature_importances_
                          feature_names = self.setup_env.X_train.columns # Use columns from processed train set
                          importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
                          feature_importance_data = importance_df.to_dict(orient='records')
                          logging.info("Extracted raw feature importances.")
                     elif hasattr(model_object, 'named_steps') and 'actual_estimator' in model_object.named_steps and hasattr(model_object.named_steps['actual_estimator'], 'feature_importances_'):
                         # Case for pipelines
                         importances = model_object.named_steps['actual_estimator'].feature_importances_
                         feature_names = self.setup_env.X_train.columns
                         importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
                         feature_importance_data = importance_df.to_dict(orient='records')
                         logging.info("Extracted raw feature importances from pipeline step.")
                     else:
                          logging.warning("Could not automatically extract raw feature importances.")

                     # Still generate and log the plot
                     self._log_plot_artifact(
                          plot_function=lambda save, verbose, **kwargs: self.pycaret_module.plot_model(model_object, plot='feature', save=save, verbose=verbose, **kwargs),
                          base_plot_name="SHAP Summary Plot", # Generic plot type name
                            model_name=model_name # Pass the specific model name
                     )
                except Exception as fi_e:
                     logging.error(f"Failed to generate/log feature importance for {model_name}: {fi_e}")

                # --- Conditionally Log SHAP Summary Plot ---
                # (Keep existing conditional SHAP logic here)
                logging.info("Checking SHAP support...")
                model_id_approx = model_name.split('_')[1] if '_' in model_name else model_name # Approximate id
                supported_shap_summary_models = {'et', 'lightgbm', 'xgboost', 'rf', 'dt', 'catboost'}
                can_run_shap = False
                if self.task_type == 'classification' and model_id_approx in supported_shap_summary_models:
                     can_run_shap = True
                     logging.info(f"Model type '{model_id_approx}' supports SHAP summary plot via interpret_model.")
                     self._log_plot_artifact(
                         plot_function=lambda save, verbose, **kwargs: self.pycaret_module.interpret_model(model_object, plot='summary', save=save), # removed verbose
                         base_plot_name="SHAP Summary Plot", # Generic plot type name
                        model_name=model_name # Pass the specific model name
                     )
                else:
                     logging.warning(f"SHAP summary plot via interpret_model() is not supported by PyCaret for model type '{model_id_approx}' in task '{self.task_type}'. Skipping SHAP summary plot generation.")
                     # Placeholder: Implement manual SHAP if needed for other models (more complex)
                     # if self.config.get("return_raw_explanations"):
                     #     logging.info("Attempting manual SHAP value calculation...")
                     #     # ... Add manual SHAP logic here ...
                     #     pass


                analysis_results["feature_importances"] = feature_importance_data
                analysis_results["shap_summary_plot_generated"] = can_run_shap
                # analysis_results["shap_values"] = shap_values_data # Add if implemented


                mlflow.set_tag("status", "completed")

        except Exception as e:
            logging.error(f"Failed during analysis of model {model_name}: {e}", exc_info=True)
            if 'nested_run' in locals() and nested_run:
                 mlflow.set_tag("status", "failed")
                 mlflow.log_param(f"error_analysis_{model_name}", str(e)[:MLFLOW_PARAM_VALUE_MAX_LEN]) # Unique error key
            return None # Indicate failure

        # Return raw data if requested and available
        if self.config.get("return_raw_explanations"):
             analysis_results["feature_importances"] = feature_importance_data # Ensure this is assigned above
             analysis_results["shap_summary_plot_generated"] = can_run_shap # Ensure this is assigned above
             return analysis_results
        else:
             return {"holdout_metrics": analysis_results.get("holdout_metrics")}

    def finalize_and_save_model(self, model_object: Any, model_name: str) -> Optional[str]:
        """Finalizes the model on the full dataset and saves artifact."""
        if self.pycaret_module is None or self.setup_env is None:
            logging.error("PyCaret environment not set up.")
            return None
        if model_object is None:
             logging.error("Model object must be provided for finalization.")
             return None

        self._start_mlflow_run()
        mlflow.set_tag("pipeline_stage", f"finalizing_{model_name}")
        logging.info(f"--- 6. Finalizing and Saving Model: {model_name} ---")

        try:
            with mlflow.start_run(run_id=self.active_mlflow_run_id, nested=True) as nested_run:
                mlflow.set_tag("finalized_model_name", model_name)

                self.final_model = self.pycaret_module.finalize_model(model_object)
                logging.info(f"Model finalized successfully.")

                # Save final model artifact locally (includes metadata saving)
                saved_model_base_path = self._save_model_artifact(self.final_model, "final", model_name)

                if saved_model_base_path:
                    # --- Generate Model Card ---
                    if self.config.get("generate_model_card"):
                        logging.info("Generating basic model card...")
                        try:
                            card_content = self._generate_model_card(model_name, saved_model_base_path)
                            card_path = f"{saved_model_base_path}_model_card.md"
                            with open(card_path, "w") as f:
                                f.write(card_content)
                            mlflow.log_artifact(card_path)
                            logging.info(f"Model card saved and logged: {card_path}")
                        except Exception as card_e:
                             logging.error(f"Failed to generate or log model card: {card_e}")

                    # --- Register Model in MLflow ---
                    if self.config.get("register_model_in_mlflow"):
                        logging.info("Registering model in MLflow Model Registry...")
                        self.register_model(
                            model_uri=f"runs:/{self.active_mlflow_run_id}/pycaret_model", # Standard path if PyCaret logged it
                            registered_model_name=self.config.get("mlflow_registered_model_name", model_name),
                            description=f"AutoML generated model: {model_name}",
                            stage=self.config.get("mlflow_model_stage", "Staging"),
                            await_registration_for=0 # Don't wait
                         )


                    mlflow.set_tag("status", "completed")
                    return f"{saved_model_base_path}.pkl" # Return full path to PKL
                else:
                     raise RuntimeError("Failed to save the finalized model artifact.")


        except Exception as e:
            logging.error(f"Failed to finalize or save model {model_name}: {e}", exc_info=True)
            if 'nested_run' in locals() and nested_run:
                 mlflow.set_tag("status", "failed")
                 mlflow.log_param("error", str(e))
            return None

# --- New Feature Methods ---

    def register_model(self, model_uri: str, registered_model_name: str, description: str = "", stage: Optional[str] = None, tags: Optional[Dict] = None, await_registration_for=300) -> Optional[mlflow.entities.model_registry.ModelVersion]:
        """Registers a model in the MLflow Model Registry."""
        logging.info(f"--- Registering Model ---")
        logging.info(f"URI: {model_uri}, Name: {registered_model_name}, Stage: {stage}")
        try:
            client = mlflow.tracking.MlflowClient()

            # Register the model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
                await_registration_for=await_registration_for,
                tags=tags
            )
            logging.info(f"Model registered: Name={model_version.name}, Version={model_version.version}")

            # Add description
            if description:
                client.update_model_version(
                    name=model_version.name,
                    version=model_version.version,
                    description=description
                )
                logging.info("Model version description updated.")

            # Transition stage if specified
            if stage in ["Staging", "Production", "Archived"]:
                client.transition_model_version_stage(
                    name=model_version.name,
                    version=model_version.version,
                    stage=stage,
                    archive_existing_versions=(stage == "Production") # Archive others if moving to Prod
                )
                logging.info(f"Model version {model_version.version} transitioned to '{stage}'.")

            return model_version

        except Exception as e:
            logging.error(f"Failed to register model {registered_model_name}: {e}", exc_info=True)
            return None

    def check_data_drift(self, new_data: pd.DataFrame, report_save_path: Optional[str] = None) -> Optional[Dict]:
        """Checks for data drift between new data and the training data."""
        if Report is None or DataDriftPreset is None:
            logging.error("Evidently library not installed. Cannot perform drift check.")
            return None
        if self.train_data is None:
            logging.error("Training data not available (was setup run?). Cannot perform drift check.")
            return None

        logging.info("--- Checking for Data Drift ---")
        if self.active_mlflow_run_id:
            self._start_mlflow_run() # Log drift check info to current run
            mlflow.set_tag("pipeline_stage", "checking_drift")

        try:
            # Ensure column order matches if necessary (Evidently usually handles by name)
            reference_data = self.train_data.copy()
            current_data = new_data.copy()
            
            logging.debug("Original reference dtypes:\n%s", reference_data.dtypes.to_string())
            logging.debug("Original current dtypes:\n%s", current_data.dtypes.to_string())
            # --- Align columns ---
            # Ensure both dataframes have the same columns in the same order for comparison
            ref_cols = list(reference_data.columns)
            cur_cols = list(current_data.columns)
            common_cols = [col for col in ref_cols if col in cur_cols]

            if set(ref_cols) != set(cur_cols):
                 logging.warning(f"Column mismatch detected between reference ({len(ref_cols)}) and current ({len(cur_cols)}) data for drift check. Using {len(common_cols)} common columns.")
                 logging.warning(f"Missing in Current: {list(set(ref_cols) - set(cur_cols))}")
                 logging.warning(f"Extra in Current: {list(set(cur_cols) - set(ref_cols))}")

            # Use only common columns IN THE ORDER OF THE REFERENCE DATA
            reference_data = reference_data[common_cols]
            current_data = current_data[common_cols]


            # --- Convert Categorical Types ---
            converted_cols_ref = []
            converted_cols_cur = []
            for col in common_cols:
                # Check reference data
                if pd.api.types.is_categorical_dtype(reference_data[col].dtype):
                    logging.debug(f"Converting categorical column '{col}' in reference data to object dtype.")
                    try:
                        # Convert to the underlying dtype if possible, otherwise object
                        original_dtype = reference_data[col].cat.categories.dtype
                        reference_data[col] = reference_data[col].astype(original_dtype)
                        converted_cols_ref.append(f"{col} (to {original_dtype})")
                    except Exception: # Fallback to object
                        reference_data[col] = reference_data[col].astype(object)
                        converted_cols_ref.append(f"{col} (to object)")

                # Check current data (might also have categoricals if loaded specially)
                if pd.api.types.is_categorical_dtype(current_data[col].dtype):
                    logging.debug(f"Converting categorical column '{col}' in current data to object dtype.")
                    try:
                        original_dtype = current_data[col].cat.categories.dtype
                        current_data[col] = current_data[col].astype(original_dtype)
                        converted_cols_cur.append(f"{col} (to {original_dtype})")
                    except Exception: # Fallback to object
                        current_data[col] = current_data[col].astype(object)
                        converted_cols_cur.append(f"{col} (to object)")

            if converted_cols_ref: logging.info(f"Converted reference columns for drift check: {converted_cols_ref}")
            if converted_cols_cur: logging.info(f"Converted current columns for drift check: {converted_cols_cur}")
            logging.debug("Converted reference dtypes:\n%s", reference_data.dtypes.to_string())
            logging.debug("Converted current dtypes:\n%s", current_data.dtypes.to_string())
            # --- End FIX ---

            drift_report = Report(metrics=[DataDriftPreset()])
            drift_report.run(current_data=current_data, reference_data=reference_data, column_mapping=None) # Auto maps columns

            drift_results = drift_report.as_dict()
            drift_detected = drift_results['metrics'][0]['result']['dataset_drift'] # Check overall drift flag
            num_drifted_features = drift_results['metrics'][0]['result']['number_of_drifted_columns']

            logging.info(f"Data drift detected: {drift_detected}")
            logging.info(f"Number of drifted features: {num_drifted_features}")
            mlflow.log_metric("data_drift_detected_flag", int(drift_detected))
            mlflow.log_metric("data_drift_num_drifted_features", num_drifted_features)

            # Save and log report if path provided
            if report_save_path:
                drift_report.save_html(report_save_path)
                logging.info(f"Drift report saved to {report_save_path}")
                mlflow.log_artifact(report_save_path)
            elif self.config.get("drift_report_name"): # Save using default name from config
                default_report_path = self.config["drift_report_name"]
                drift_report.save_html(default_report_path)
                logging.info(f"Drift report saved to {default_report_path}")
                mlflow.log_artifact(default_report_path)


            return {
                "drift_detected": drift_detected,
                "num_drifted_features": num_drifted_features,
                "metrics": drift_results['metrics'] # Return detailed metrics
            }

        except Exception as e:
            logging.error(f"Data drift check failed: {e}", exc_info=True)
            mlflow.set_tag("drift_check_status", "failed")
            return None

    def generate_run_report(self, mlflow_run_id: str):
        """Generates a basic summary report for a given MLflow run."""
        logging.warning("Automated Run Report Generation is not fully implemented.")
        logging.info(f"--- Generating Report for Run ID: {mlflow_run_id} ---")
        # Placeholder:
        # 1. Use mlflow client to get run data (params, metrics, tags, artifacts list)
        # 2. Format data into a string (Markdown or HTML)
        # 3. Optionally render HTML/PDF using Jinja2/Weasyprint
        # 4. Save and potentially log the report file
        try:
            client = mlflow.tracking.MlflowClient()
            run_data = client.get_run(mlflow_run_id).to_dictionary()
            report_content = f"# AutoML Run Summary: {mlflow_run_id}\n\n"
            report_content += "## Parameters:\n```json\n" + json.dumps(run_data.get('data', {}).get('params', {}), indent=2) + "\n```\n\n"
            report_content += "## Metrics:\n```json\n" + json.dumps(run_data.get('data', {}).get('metrics', {}), indent=2) + "\n```\n\n"
            report_content += "## Tags:\n```json\n" + json.dumps(run_data.get('data', {}).get('tags', {}), indent=2) + "\n```\n\n"
            # List artifacts (optional)
            # artifacts = client.list_artifacts(mlflow_run_id)
            # report_content += "## Artifacts:\n" + "\n".join([f"- {a.path}" for a in artifacts]) + "\n"

            report_path = f"automl_run_{mlflow_run_id}_report.md"
            with open(report_path, "w") as f:
                f.write(report_content)
            logging.info(f"Basic Markdown report saved to {report_path}")
            # You could log this report back to the *same* run or a different location
            # mlflow.log_artifact(report_path)
            return report_path
        except Exception as e:
            logging.error(f"Failed to generate basic run report: {e}", exc_info=True)
            return None

    def explain_prediction(self, model_base_path: str, data_instance: pd.DataFrame) -> Optional[Dict]:
        """Explains a single prediction using SHAP."""
        if shap is None:
            logging.error("SHAP library not installed. Cannot explain prediction.")
            return None

        logging.info("--- Explaining Single Prediction ---")

        # 1. Load Model and get task type
        model_object, task_type = self._load_model_and_metadata(model_base_path)
        if model_object is None:
            return None

        # --- FIX: Separate Preprocessor and Estimator from LOADED model ---
        final_estimator = None
        preprocessor_pipeline = None
        transformed_instance = None

        if hasattr(model_object, 'steps'): # Check if it's a scikit-learn Pipeline
            try:
                if len(model_object.steps) > 1:
                    # Create a new pipeline containing all steps EXCEPT the last one
                    preprocessor_pipeline = Pipeline(model_object.steps[:-1])
                    # Extract the final estimator object (it's the second item in the step tuple)
                    final_estimator = model_object.steps[-1][1]
                    logging.info(f"Successfully separated preprocessor and final estimator ({type(final_estimator).__name__}).")
                elif len(model_object.steps) == 1: # Pipeline with only the model (no preprocessing?)
                     logging.warning("Loaded pipeline has only one step. Assuming it's the final estimator.")
                     final_estimator = model_object.steps[0][1]
                     preprocessor_pipeline = None # No preprocessing steps
                else: # Empty pipeline?
                     logging.error("Loaded pipeline has no steps.")
                     return None

            except Exception as e:
                logging.error(f"Failed to reconstruct preprocessor pipeline or extract estimator: {e}", exc_info=True)
                return None
        else:
            # If the loaded object isn't a pipeline, assume it's just the estimator
            logging.warning("Loaded model object is not a scikit-learn Pipeline. Assuming it's the final estimator with no preprocessing.")
            final_estimator = model_object
            preprocessor_pipeline = None

        # 2. Preprocess the single instance using the extracted steps (if any)
        try:
            # Ensure data_instance is a DataFrame
            if not isinstance(data_instance, pd.DataFrame):
                data_instance = pd.DataFrame([data_instance]) # Assume dict or Series

            if preprocessor_pipeline:
                logging.info("Applying preprocessing steps to data instance...")
                # Ensure columns match what pipeline expects - requires user to pass correct df
                transformed_instance = preprocessor_pipeline.transform(data_instance)
                logging.info("Preprocessing complete.")
            else:
                logging.info("No preprocessing steps found to apply.")
                transformed_instance = data_instance # Use original data if no preprocessor

        except Exception as e:
            logging.error(f"Failed to apply preprocessing pipeline to data instance: {e}", exc_info=True)
            return None
        # --- End FIX ---


        # 3. Initialize SHAP Explainer based on model type
        if final_estimator is None:
             logging.error("Final estimator could not be determined.")
             return None
        if transformed_instance is None:
             logging.error("Transformed instance data is not available.")
             return None

        try:
            logging.info(f"Initializing SHAP explainer for model type: {type(final_estimator).__name__}")

            # Convert sparse matrix to dense if needed by explainer/model type
            # Some SHAP explainers work better with dense data
            if hasattr(transformed_instance, "toarray"):
                try:
                     transformed_instance_dense = transformed_instance.toarray()
                except Exception as dense_err:
                     logging.warning(f"Could not convert transformed data to dense array: {dense_err}. Proceeding with original.")
                     transformed_instance_dense = transformed_instance # Use original sparse if fails
            else:
                transformed_instance_dense = transformed_instance

            # Choose appropriate SHAP explainer
            # Note: Background data (like self.train_data transformed) might be needed for some explainers (e.g., Kernel)
            # For simplicity, we start with shap.Explainer which tries to auto-detect.
            # Background data might be needed for KernelExplainer for better results.
            # background_data = shap.sample(transformed_training_data, 100) # Needs transformed training data
            explainer = shap.Explainer(final_estimator, transformed_instance_dense) # Pass dense version
            logging.info(f"Using SHAP explainer: {type(explainer)}")


            # 4. Calculate SHAP values for the instance(s)
            # Use the dense data for calculation as well if converted
            shap_values = explainer(transformed_instance_dense)
            logging.info("SHAP values calculated.")

            # 5. Format Output
            base_value = explainer.expected_value
            # Handle multi-output (e.g., classification probabilities) for base_value and shap_values
            if isinstance(base_value, (list, np.ndarray)): base_value = base_value[0] # Example: Take first output's base value
            if isinstance(shap_values.values, (list, np.ndarray)) and len(shap_values.values.shape) > 2:
                 # Multi-class output: shap_values.values shape might be (n_samples, n_features, n_classes)
                 # Take values for the first sample, and perhaps the class with highest probability? Or sum across classes?
                 # Let's take values for the first output/class for simplicity here.
                 instance_shap_values = shap_values.values[0, :, 0] # SHAP values for sample 0, class 0
                 logging.warning("SHAP values appear multi-output; showing explanation for the first output class.")
            else:
                  instance_shap_values = shap_values.values[0] # Values for the first (only) instance/output

            # Feature names might come from shap_values object or need to be inferred
            feature_names = shap_values.feature_names
            if feature_names is None:
                 # Fallback if SHAP couldn't get names (e.g., from DataFrame)
                 if isinstance(transformed_instance_dense, pd.DataFrame):
                      feature_names = transformed_instance_dense.columns.tolist()
                 else: # Try getting from original setup if available (less reliable here)
                      try:
                         feature_names = self.setup_env.X_train.columns.tolist()
                      except:
                          feature_names = [f"feature_{i}" for i in range(transformed_instance_dense.shape[1])]


            explanation = {
                 "base_value": base_value.tolist() if hasattr(base_value, 'tolist') else base_value,
                 "shap_values": instance_shap_values.tolist() if hasattr(instance_shap_values, 'tolist') else instance_shap_values,
                 "feature_names": feature_names,
                 "data_instance": data_instance.iloc[0].to_dict() # Original untransformed data
            }

            return explanation

        except Exception as e:
            logging.error(f"Failed during SHAP explanation calculation: {e}", exc_info=True)
            return None

    def predict_on_new_data(self, new_data: pd.DataFrame, model_base_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Makes predictions on new data using a saved model pipeline.
        Loads model and necessary metadata (task type).
        """
        logging.info("--- 7. Making Predictions ---")

        load_path = model_base_path
        # Find latest model if path not provided
        if load_path is None:
            try:
                final_models = [f.replace('.pkl','') for f in os.listdir(self.config["model_save_dir"]) if f.startswith('final_') and f.endswith('.pkl')]
                if not final_models:
                    logging.error(f"No final models found in '{self.config['model_save_dir']}'.")
                    return None
                # Find file with latest modification time based on the PKL file
                latest_final_model_base = max(final_models, key=lambda f_base: os.path.getmtime(os.path.join(self.config["model_save_dir"], f"{f_base}.pkl")))
                load_path = os.path.join(self.config["model_save_dir"], latest_final_model_base)
                logging.info(f"Using latest final model: {load_path}")
            except Exception as e:
                logging.error(f"Error finding latest model: {e}", exc_info=True)
                return None
        else:
            logging.info(f"Using specified model base path: {load_path}")

        # Load model and its metadata (task type)
        loaded_pipeline, loaded_task_type = self._load_model_and_metadata(load_path)

        if loaded_pipeline is None:
            logging.error("Failed to load the model pipeline.")
            return None
        if loaded_task_type is None:
            logging.warning("Task type could not be determined from metadata. Prediction behaviour might be unexpected.")
            # Use self.task_type as a fallback if available from the current instance run
            loaded_task_type = self.task_type if self.task_type else 'classification' # Last resort guess


        # Determine the correct PyCaret module
        pred_module = pyclf if loaded_task_type == 'classification' else pyreg

        # Optional: Data Drift Check before prediction
        if self.config.get("enable_drift_check_on_predict"):
            logging.info("Performing data drift check before prediction...")
            # Assume profile was saved during setup (self.training_data_profile_path)
            # Or load reference data (self.train_data) if available
            drift_results = self.check_data_drift(new_data=new_data) # Uses self.train_data internally
            if drift_results and drift_results.get("drift_detected"):
                logging.warning("Potential data drift detected! Prediction results may be unreliable.")
                # Optionally: Add logic here to halt prediction or use a different model based on drift

        try:
            logging.info(f"Generating predictions using {pred_module.__name__}...")
            predictions = pred_module.predict_model(loaded_pipeline, data=new_data, verbose=False)
            logging.info("Predictions generated successfully.")
            return predictions

        except Exception as e:
            logging.error(f"An error occurred during prediction: {e}", exc_info=True)
            return None

# **4. Updated `if __name__ == "__main__":` block (Example Usage):**

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("="*50)
    logging.info("Starting Enhanced AutoML Runner Script")
    logging.info(f"PyCaret Version: {pycaret.__version__}")
    logging.info(f"MLflow Version: {mlflow.__version__}")
    logging.info("="*50)

    # 1. Initialize the runner
    runner = AutoMLRunner(config=CONFIG)
    final_model_saved_path = None
    analysis_results = None
    tuned_model_result = None
    ensemble_model_result = None

    try:
        # 2. Setup Environment
        success, task_type = runner.setup_automl_environment(
            data_path=CONFIG["data_file_path"],
            target_column=CONFIG["target_column"],
            feature_columns=CONFIG.get("feature_columns") # Pass optional feature list
        )
        if not success:
            raise RuntimeError("Failed to setup AutoML environment.")

        # 3. Compare Models & Analyze Baselines
        compare_results = runner.compare_models_and_baselines()
        if compare_results is None:
            raise RuntimeError("Failed during model comparison or baseline analysis.")
        best_model_id = compare_results.get("best_model_id")
        logging.info(f"Comparison complete. Best model type: {best_model_id}")

        # 4. Tune a Model (Example: Tune the best model found)
        if best_model_id:
            tuned_model_result = runner.tune_model(use_best_from_compare=True)
            if tuned_model_result:
                logging.info(f"Successfully tuned model: {tuned_model_result.get('tuned_model_id')}")
                # runner.tuned_model holds the object if needed later
            else:
                logging.warning(f"Failed to tune the best model {best_model_id}.")
        else:
             logging.warning("Skipping tuning as no best model was identified.")
             # Optionally, try tuning a default model:
             # tuned_model_result = runner.tune_model(model_id='rf') # Example

        # --- Choose Model for Final Steps (Tuned or Best Untuned) ---
        model_to_analyze_finalize = None
        model_to_analyze_finalize_name = None
        # Proceed with TUNED model
        if tuned_model_result and tuned_model_result.get("tuned_model_object"):
            model_to_analyze_finalize = tuned_model_result["tuned_model_object"]
            model_to_analyze_finalize_name = f"tuned_{tuned_model_result.get('tuned_model_id', 'model')}"
            logging.info(f"Proceeding with TUNED model: {model_to_analyze_finalize_name}")

        elif runner.best_model_obj:
             model_to_analyze_finalize = runner.best_model_obj
             model_to_analyze_finalize_name = f"best_untuned_{best_model_id or 'model'}"
             logging.warning(f"Proceeding with BEST UNTUNED model: {model_to_analyze_finalize_name}")
        else:
            logging.error("No suitable model available to analyze or finalize.")
            # Optionally try a baseline: runner.baseline_models_trained.get('rf')

        # 5. Analyze the chosen model (Tuned or Best Untuned)
        if model_to_analyze_finalize:
            analysis_results = runner.analyze_trained_model(
                model_object=model_to_analyze_finalize,
                model_name=model_to_analyze_finalize_name
            )
            if analysis_results:
                logging.info(f"Analysis complete for {model_to_analyze_finalize_name}.")
                # Store metrics if needed for model card
                runner.latest_analysis_metrics = analysis_results.get("holdout_metrics")
            else:
                 logging.warning(f"Analysis failed for {model_to_analyze_finalize_name}")

        # 6. Create Ensemble (Optional - Example using the model chosen above)
        # if model_to_analyze_finalize:
        #     ensemble_model_result = runner.create_ensemble_model(
        #         base_model_object=model_to_analyze_finalize,
        #         base_model_id=model_to_analyze_finalize_name.replace('tuned_','').replace('best_untuned_','') # Approx ID
        #     )
        #     if ensemble_model_result:
        #         logging.info(f"Successfully created ensemble model: {ensemble_model_result.get('ensemble_model_id')}")
        #         # If ensembling worked, maybe analyze and finalize IT instead:
        #         model_to_analyze_finalize = ensemble_model_result["ensemble_model_object"]
        #         model_to_analyze_finalize_name = ensemble_model_result.get('ensemble_model_id', 'ensemble')
        #         # Re-run analysis on the ensemble model
        #         analysis_results = runner.analyze_trained_model(model_to_analyze_finalize, model_to_analyze_finalize_name)
        #         # ... store metrics ...
        #     else:
        #         logging.warning("Failed to create ensemble model.")

        # 7. Finalize and Save the chosen model (and potentially register)
        if model_to_analyze_finalize:
            final_model_saved_path = runner.finalize_and_save_model(
                model_object=model_to_analyze_finalize,
                model_name=model_to_analyze_finalize_name # Use the descriptive name
            )
            if final_model_saved_path:
                logging.info(f"Final model saved successfully: {final_model_saved_path}")
            else:
                raise RuntimeError("Failed to finalize and save the model.")
        else:
            logging.error("No model was finalized or saved.")


        # --- Post-Training Operations ---

        # 8. Prediction Example
        logging.info("\n--- Running Prediction Example ---")
        if runner.data is not None and final_model_saved_path:
            try:
                # Create Dummy Data (same logic as before)
                # feature_columns = runner.data.drop(columns=[CONFIG["target_column"]]).columns
                # example_row = {}
                # original_data_for_types = runner.data.drop(columns=[CONFIG["target_column"]])
                # for col in feature_columns:
                #     col_dtype = original_data_for_types[col].dtype
                #     if pd.api.types.is_numeric_dtype(col_dtype):
                #         example_row[col] = [original_data_for_types[col].mean()]
                #     else:
                #         mode_val = original_data_for_types[col].mode()
                #         example_row[col] = [mode_val[0]] if not mode_val.empty else [None]
                # new_data_example = pd.DataFrame(example_row)
                new_data_example = pd.read_csv ("E:/Grab Bootcamp/Grab-Tech-Bootcamp/ML/automl/test.csv")
                logging.info("Example new data row for prediction:")
                print(new_data_example)

                # Predict using the specific finalized model path (without .pkl)
                predictions_df = runner.predict_on_new_data(
                    new_data=new_data_example,
                    model_base_path=final_model_saved_path.replace('.pkl','')
                )

                if predictions_df is not None:
                    logging.info("Predictions on new data:")
                    print(predictions_df)

                    # 9. Explain Prediction Example (if enabled and prediction worked)
                    if CONFIG.get("enable_prediction_explanation") and not predictions_df.empty:
                         explanation = runner.explain_prediction(
                             model_base_path=final_model_saved_path.replace('.pkl',''),
                             data_instance=new_data_example.iloc[[0]] # Pass first row as DataFrame
                         )
                         if explanation:
                              logging.info("\n--- Prediction Explanation (SHAP) ---")
                              print(f"Base Value: {explanation['base_value']}")
                              # Print SHAP values for features contributing most
                              shap_series = pd.Series(explanation['shap_values'], index=explanation['feature_names']).sort_values(ascending=False)
                              print("Top contributing features (SHAP values):")
                              print(shap_series.head())
                              print(shap_series.tail())
                         else:
                              logging.warning("Could not generate prediction explanation.")

                else:
                    logging.warning("Prediction example failed.")

            except Exception as pred_ex:
                logging.error(f"Failed to run prediction/explanation example: {pred_ex}", exc_info=True)
        else:
            logging.warning("Skipping prediction example (no data or final model path).")


        # 10. Generate Run Report (Example)
        if runner.active_mlflow_run_id:
             logging.info("\n--- Generating Run Report ---")
             runner.generate_run_report(runner.active_mlflow_run_id)


    except Exception as pipeline_ex:
         logging.error(f"Pipeline execution failed: {pipeline_ex}", exc_info=True)
         # Ensure MLflow run is ended on error
         runner._end_mlflow_run(status="FAILED") # Use internal method safely
    finally:
        # Ensure MLflow run is ended if script finishes potentially before pipeline logic ends it
        runner._end_mlflow_run(status="FINISHED") # Will only end if still running
        logging.info("="*50)
        logging.info("Enhanced AutoML Runner Script Finished")
        logging.info("="*50)