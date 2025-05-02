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
import yaml # For loading config
import json # For saving task_type metadata
from typing import List, Optional, Dict, Any, Tuple, Union
import pycaret # Keep for version check if needed elsewhere

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

# --- Helper Functions ---

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    logging.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at '{config_path}'")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration: {e}", exc_info=True)
        raise

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns to smallest possible type that fits the data range
    and convert object columns with low cardinality to 'category'.
    Handles NaN/NA values by using Pandas nullable integer types where appropriate
    and skipping comparisons if min/max return NA.
    Reduces memory usage significantly.
    """
    logging.info("Optimizing DataFrame dtypes for memory efficiency...")
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        numerics = pd.api.types.is_numeric_dtype(col_type)
        contains_na = df[col].isnull().any() # Check for NaN/NA values

        if numerics:
            # Calculate min/max, skipping NA values for the calculation itself
            try:
                c_min = df[col].min(skipna=True)
                c_max = df[col].max(skipna=True)
            except TypeError:
                # Handle cases like all-NA columns for specific dtypes if min/max fail even with skipna
                logging.warning(f"Could not compute min/max for column '{col}' (possibly all NA or mixed types). Skipping numeric optimization.")
                continue # Skip optimization for this column

            # --- Check if min/max results are NA ---
            # If the column consisted *only* of NA, min/max might return NaN or NaT
            # We also check explicitly for pd.NA which is the result for nullable int/bool types
            if pd.isna(c_min) or pd.isna(c_max):
                logging.debug(f"Column '{col}' contains only NA or non-comparable values. Skipping numeric downcasting.")
                continue # Skip numeric optimization for this column

            # --- Proceed with checks only if min/max are valid numbers ---
            # Integer type checks
            # Check if float contains only whole numbers (handle potential NA during check)
            is_integer_like_float = (col_type.kind == 'f' and pd.Series(np.mod(df[col].dropna(), 1) == 0).all())

            if col_type.kind == 'i' or col_type.kind == 'u' or is_integer_like_float:
                # Determine target integer type based on valid range
                try: # Add try-except around comparisons for extra safety
                    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                        target_type = pd.Int8Dtype() if contains_na else np.int8
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                        target_type = pd.Int16Dtype() if contains_na else np.int16
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                        target_type = pd.Int32Dtype() if contains_na else np.int32
                    else: # Default to 64-bit
                        target_type = pd.Int64Dtype() if contains_na else np.int64

                    # Apply the conversion only if it's different from current type
                    if df[col].dtype != target_type:
                         try:
                             df[col] = df[col].astype(target_type)
                             # No need for debug log here, done previously
                         except Exception as e:
                              logging.warning(f"Could not convert column '{col}' to {target_type}. Keeping original type {df[col].dtype}. Error: {e}")

                except TypeError as te:
                     logging.warning(f"TypeError during range comparison for column '{col}'. Keeping original type {df[col].dtype}. Error: {te}")


            # Float type checks (only if not already handled as integer-like)
            elif col_type.kind == 'f':
                 try: # Add try-except around comparisons for extra safety
                     # Check if float32 is sufficient
                     if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                         if df[col].dtype != np.float32:
                              df[col] = df[col].astype(np.float32)
                     # else: keep as float64 (already checked)
                 except TypeError as te:
                     logging.warning(f"TypeError during float range comparison for column '{col}'. Keeping original type {df[col].dtype}. Error: {te}")


        # Object/Category type checks
        elif col_type == 'object':
            # Convert low cardinality objects to category
            try:
                # Ensure nunique() doesn't fail on weird data
                num_unique = df[col].nunique()
                if num_unique / len(df[col]) < 0.5:
                    if df[col].dtype != 'category':
                        df[col] = df[col].astype('category')
            except Exception as e:
                 logging.warning(f"Could not check uniqueness or convert column '{col}' to category. Error: {e}")


    end_mem = df.memory_usage().sum() / 1024**2
    # Ensure start_mem is not zero before division
    if start_mem > 0:
        mem_reduction_pct = (start_mem - end_mem) / start_mem * 100
        logging.info(f"Memory usage after optimization: {end_mem:.2f} MB ({mem_reduction_pct:.1f}% reduction from {start_mem:.2f} MB).")
    else:
         logging.info(f"Memory usage after optimization: {end_mem:.2f} MB (Initial memory was zero).")

    return df

# --- AutoML Runner Class ---
class AutoMLRunner:
    """
    Encapsulates the AutoML workflow using PyCaret and MLflow,
    optimized for larger datasets.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.task_type: Optional[str] = None
        self.pycaret_module: Optional[Any] = None
        self.sort_metric: Optional[str] = None
        self.optimize_metric: Optional[str] = None
        self.setup_env: Optional[Any] = None
        self.best_model_obj: Optional[Any] = None
        self.results_df: Optional[pd.DataFrame] = None
        self.baseline_models_trained: Dict[str, Any] = {}
        self.tuned_best_model: Optional[Any] = None
        self.final_model: Optional[Any] = None
        self.target_col = self.config["target_column"]
        self.n_jobs = self.config.get("n_jobs", -1) # Get n_jobs from config, default to -1

        os.makedirs(self.config["model_save_dir"], exist_ok=True)
        self._setup_mlflow()

    def _setup_mlflow(self):
        logging.info("Setting up MLflow Tracking...")
        mlflow.set_tracking_uri(self.config["mlflow_tracking_uri"])
        mlflow.set_experiment(self.config["experiment_name"])
        logging.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        logging.info(f"MLflow experiment set to: {self.config['experiment_name']}")

    def load_data(self) -> bool:
        logging.info("Loading Data...")
        try:
            # Load full data
            full_data = pd.read_csv(self.config["data_file_path"])
            logging.info(f"Full data loaded with shape: {full_data.shape}")

            if self.target_col not in full_data.columns:
                raise ValueError(f"Target column '{self.target_col}' not found.")

            # Optimize Dtypes for memory efficiency
            if self.config.get('optimize_pandas_dtypes', False):
                 full_data = optimize_dtypes(full_data)

            # --- MANUAL SAMPLING Step (if configured) ---
            if self.config.get("use_sampling_in_setup", False):
                logging.info("Applying manual sampling before PyCaret setup...")
                sample_frac = self.config.get("sample_fraction", None)
                sample_n = self.config.get("sample_n", None)
                random_state = self.config.get("session_id", None) # Use session_id for reproducibility

                if sample_n is not None:
                    # Ensure sample_n is not larger than dataset size
                    sample_n = min(sample_n, len(full_data))
                    logging.info(f"Sampling {sample_n} rows (random_state={random_state}).")
                    self.data = full_data.sample(n=sample_n, random_state=random_state)
                elif sample_frac is not None:
                    logging.info(f"Sampling fraction {sample_frac} (random_state={random_state}).")
                    self.data = full_data.sample(frac=sample_frac, random_state=random_state)
                else:
                    logging.warning("`use_sampling_in_setup` is True, but neither `sample_fraction` nor `sample_n` provided. Using full data.")
                    self.data = full_data # Fallback to full data
            else:
                 logging.info("Using full dataset for PyCaret setup (sampling disabled).")
                 self.data = full_data # Assign full data if sampling is off

            logging.info(f"Data prepared for PyCaret setup with shape: {self.data.shape}")
            self.data = self.data.reset_index(drop=True)
            # Log sampling status to MLflow here if possible, or pass flag to run_pipeline
            self._manual_sampling_applied = self.config.get("use_sampling_in_setup", False) and (sample_n is not None or sample_frac is not None)

            return True
        except FileNotFoundError:
            logging.error(f"Data file not found at '{self.config['data_file_path']}'")
            return False
        except Exception as e:
            logging.error(f"Error loading or sampling data: {e}", exc_info=True)
            return False
        
    def detect_task_type(self) -> bool:
        if self.data is None:
            logging.error("Data not loaded. Cannot detect task type.")
            return False
        logging.info("Auto-Detecting Task Type...")
        target_series = self.data[self.target_col]
        n_unique = target_series.nunique()
        dtype = target_series.dtype

        # --- Task Detection Logic (same as before, generally robust) ---
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
            self.task_type = 'classification'
        elif pd.api.types.is_bool_dtype(dtype):
             self.task_type = 'classification'
        elif pd.api.types.is_numeric_dtype(dtype):
            unique_values = target_series.unique()
            if len(unique_values) == 2 and np.all(np.isin(unique_values, [0, 1])):
                self.task_type = 'classification'
                logging.info("Detected task type: Binary Classification")
            elif n_unique < self.config["unique_value_threshold_for_classification"]:
                self.task_type = 'classification'
                logging.info(f"Detected task type: Classification (Numeric, {n_unique} unique < threshold)")
                if pd.api.types.is_float_dtype(dtype):
                     logging.warning("Target is float but has few unique values. Treating as classification.")
            else:
                self.task_type = 'regression'
                logging.info(f"Detected task type: Regression (Numeric, {n_unique} unique >= threshold)")
        else:
            logging.warning(f"Could not reliably determine task type for dtype {dtype}. Defaulting to classification.")
            self.task_type = 'classification'

        logging.info(f"Final Detected Task Type: {self.task_type.upper()}")
        return True

    def setup_pycaret(self) -> bool:
        if self.data is None or self.task_type is None:
            # Now self.data might be the sampled data
            logging.error("Data (potentially sampled) or task type not available for PyCaret setup.")
            return False
        logging.info("Setting up PyCaret Environment on prepared data...")
        start_time = time.time()

        # Parameters for pycaret.setup - NO sample_kwargs here
        setup_params = {
            "data": self.data, # Pass the potentially sampled data
            "target": self.target_col,
            "session_id": self.config["session_id"],
            "log_experiment": True,
            "experiment_name": self.config["experiment_name"],
            "numeric_imputation": self.config["numeric_imputation"],
            "categorical_imputation": self.config.get("categorical_imputation", "mode"),
            "n_jobs": self.n_jobs,
            "fold_strategy": self.config.get("fold_strategy", "stratifiedkfold" if self.task_type == "classification" else "kfold"),
            "fold": self.config.get("baseline_folds", 5),
            # Default PyCaret sampling (for train/test split) is usually True, control via train_size if needed
            # "sampling": True, # This is usually default
            # "train_size": 0.7, # Default train_size
        }

        # --- REMOVED sample_kwargs logic ---

        if self.task_type == 'classification':
            self.pycaret_module = pyclf
            self.sort_metric = self.config["sort_metric_classification"]
            self.optimize_metric = self.config["optimize_metric_classification"]
        elif self.task_type == 'regression':
            self.pycaret_module = pyreg
            self.sort_metric = self.config["sort_metric_regression"]
            self.optimize_metric = self.config["optimize_metric_regression"]
            setup_params["normalize"] = self.config.get("normalize_regression", True)
        else:
            logging.error(f"Invalid task type determined: {self.task_type}")
            return False

        logging.info(f"PyCaret Module: {self.pycaret_module.__name__}")
        logging.info(f"Sort Metric: {self.sort_metric}, Optimize Metric: {self.optimize_metric}")
        logging.info(f"Setup Parameters (subset): n_jobs={setup_params.get('n_jobs', 'Default')}, input_data_shape={self.data.shape}") # Show shape of data going into setup

        try:
            self.setup_env = self.pycaret_module.setup(**setup_params)
            logging.info(f"PyCaret setup completed in {time.time() - start_time:.2f} seconds.")
            # Log relevant params actually used
            mlflow.log_param("pycaret_manual_sampling_applied", getattr(self, '_manual_sampling_applied', False)) # Log if manual sampling was done before setup
            mlflow.log_param("pycaret_setup_input_rows", self.data.shape[0])
            mlflow.log_param("pycaret_n_jobs", setup_params.get("n_jobs", "Default"))
            mlflow.log_param("pycaret_fold_strategy", setup_params.get("fold_strategy", "Default"))
            mlflow.log_param("pycaret_baseline_folds", setup_params.get("fold", "Default"))
            return True
        except Exception as e:
            logging.error(f"PyCaret setup failed: {e}", exc_info=True)
            return False
        
    def compare_models(self) -> bool:
        if self.pycaret_module is None or self.sort_metric is None:
            logging.error("PyCaret environment not set up for model comparison.")
            return False
        logging.info("Comparing Models...")
        start_time = time.time()

        compare_params = {
            "sort": self.sort_metric,
            # "n_jobs": self.n_jobs, # <--- REMOVE THIS LINE
            "fold": self.config.get("baseline_folds", 5),
            "include": self.config.get("include_models_compare", None), # Limit models for speed
            "exclude": self.config.get("exclude_models_compare", None)
        }
        # Log n_jobs from setup for clarity, as compare_models will use it
        logging.info(f"Compare Models Params: include={compare_params.get('include')}, exclude={compare_params.get('exclude')}, folds={compare_params.get('fold')}. Using n_jobs={self.n_jobs} (set during setup).")

        try:
            self.best_model_obj = self.pycaret_module.compare_models(**compare_params)
            self.results_df = self.pycaret_module.pull()
            # Check if results_df is valid before accessing
            if self.results_df is None or self.results_df.empty:
                 logging.warning("compare_models did not return a results dataframe.")
                 # Handle cases where compare_models might fail internally or return None
                 # Decide if this is a critical failure or if the pipeline can continue
                 # For now, we'll log and try to proceed, but may need adjustment
                 return False # Treat empty results as failure for now

            logging.info("\nTop 5 Model Comparison Results (Cross-Validated):")
            print(self.results_df.head()) # Print head for console readability
            logging.info(f"Model comparison completed in {time.time() - start_time:.2f} seconds.")

            best_model_id = self.results_df.index[0]
            logging.info(f"Best model found by compare_models: {best_model_id}")


            # Log results table to MLflow
            results_html = self.results_df.to_html(escape=False)
            mlflow.log_text(results_html, "compare_models_results.html")
            # Log metric value safely
            if self.sort_metric in self.results_df.columns:
                mlflow.log_metric(f"best_model_cv_{self.sort_metric}", self.results_df.iloc[0][self.sort_metric])
            else:
                logging.warning(f"Sort metric '{self.sort_metric}' not found in compare_models results columns.")

            return True
        except Exception as e:
            logging.error(f"Model comparison failed: {e}", exc_info=True)
            return False
        
    def _log_plot_artifact(self, plot_function: callable, plot_name: str, **kwargs):
        """Helper to generate, save, and log a plot artifact."""
        # This function remains largely the same, but logging is improved
        logging.info(f"Attempting to generate plot: '{plot_name}'")
        plot_save_path = None
        try:
            # Call the plot function - use verbose=False to avoid printing path to console
            plot_save_path = plot_function(save=True, verbose=False, **kwargs)
            if plot_save_path and os.path.exists(plot_save_path):
                try:
                    mlflow.log_artifact(plot_save_path)
                    logging.info(f"Successfully logged plot artifact: {plot_name} ({os.path.basename(plot_save_path)})")
                except Exception as log_e:
                    logging.error(f"Could not log plot artifact {plot_save_path}: {log_e}", exc_info=True)
            else:
                # PyCaret might return None or invalid path if plot fails internally
                logging.warning(f"Plot file path invalid or not generated for '{plot_name}'. Expected path might be: {plot_save_path}")
        except TypeError as te: # Catch errors if plot type isn't supported
             logging.warning(f"Could not generate plot '{plot_name}'. Model type might not support this plot or parameters incorrect. Error: {te}")
        except Exception as plot_e:
            logging.error(f"Failed to generate or save plot '{plot_name}': {plot_e}", exc_info=True)


    def _save_model_artifact(self, model_object: Any, model_stage: str, model_id: str) -> Optional[str]:
        """Helper to save a model artifact and its task_type metadata."""
        if self.pycaret_module is None or self.task_type is None:
             logging.error("Cannot save model artifact - PyCaret module or task_type unknown.")
             return None
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename_base = f"{model_stage}_{model_id}_{timestamp}"
            save_path_base = os.path.join(self.config["model_save_dir"], model_filename_base)

            # Save the model pipeline using PyCaret
            self.pycaret_module.save_model(model_object, save_path_base)
            full_save_path_pkl = f"{save_path_base}.pkl" # save_model adds .pkl

            # Save metadata (task_type) alongside the model
            metadata_path = f"{save_path_base}_meta.json"
            metadata = {"task_type": self.task_type}
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            logging.info(f"Model '{model_id}' ({model_stage}) saved to {full_save_path_pkl}")
            logging.info(f"Metadata saved to {metadata_path}")

            # Log paths as MLflow parameters/tags for traceability
            mlflow.log_param(f"{model_stage}_{model_id}_save_path_pkl", full_save_path_pkl)
            mlflow.log_param(f"{model_stage}_{model_id}_save_path_meta", metadata_path)

            # Log the metadata file itself as an artifact
            mlflow.log_artifact(metadata_path)
            # Note: PyCaret's log_experiment=True *might* automatically log the .pkl model artifact.
            # Logging it explicitly might create duplicates, but ensures it's logged if PyCaret fails to.
            # Consider logging it conditionally if needed.
            # mlflow.log_artifact(full_save_path_pkl, artifact_path=f"models/{model_stage}")


            return save_path_base # Return base path without extension

        except Exception as e:
            logging.error(f"Failed to save model artifact {model_stage} {model_id}: {e}", exc_info=True)
            return None


    def analyze_baseline_models(self) -> bool:
        if self.pycaret_module is None:
             logging.error("PyCaret module not defined for baseline analysis.")
             return False

        baseline_ids_key = f"baseline_{self.task_type}_models"
        baseline_model_ids = self.config.get(baseline_ids_key, [])

        if not baseline_model_ids:
             logging.warning(f"No baseline models specified in config key '{baseline_ids_key}'. Skipping baseline analysis.")
             return True # Not an error, just nothing to do

        logging.info(f"--- Analyzing Baseline Models: {baseline_model_ids} ---")
        success_count = 0
        for baseline_id in baseline_model_ids:
            logging.info(f"Processing baseline model: {baseline_id}")
            try:
                with mlflow.start_run(run_name=f"Baseline_{baseline_id}", nested=True) as baseline_run:
                    start_time_model = time.time()
                    mlflow.set_tag("model_stage", "baseline")
                    mlflow.set_tag("model_id", baseline_id)

                    # Create baseline model (trained on CV folds)
                    baseline_model = self.pycaret_module.create_model(
                        baseline_id,
                        fold=self.config.get("baseline_folds", 5),
                        n_jobs=self.n_jobs,
                        verbose=False # Less console output during creation
                    )
                    self.baseline_models_trained[baseline_id] = baseline_model
                    logging.info(f"Created baseline model '{baseline_id}' in {time.time() - start_time_model:.2f}s")

                    # Evaluate (usually creates plots in interactive, logs metrics via log_experiment)
                    # We might not need evaluate_model if create_model logged sufficiently via log_experiment
                    # eval_results = self.pycaret_module.evaluate_model(baseline_model) # Displays plots if interactive
                    # logging.info(f"Evaluation metrics logged for {baseline_id}")

                    # Log specific plots we want as artifacts
                    self._log_plot_artifact(
                        plot_function=lambda save, verbose: self.pycaret_module.plot_model(baseline_model, plot='feature', save=save, verbose=verbose),
                        plot_name=f"Feature Importance (Baseline {baseline_id})"
                    )
                    # Add more plots if needed (e.g., 'auc', 'confusion_matrix' for clf)
                    if self.task_type == 'classification':
                         self._log_plot_artifact(
                             plot_function=lambda save, verbose: self.pycaret_module.plot_model(baseline_model, plot='auc', save=save, verbose=verbose),
                             plot_name=f"AUC (Baseline {baseline_id})"
                         )
                         self._log_plot_artifact(
                             plot_function=lambda save, verbose: self.pycaret_module.plot_model(baseline_model, plot='confusion_matrix', save=save, verbose=verbose),
                             plot_name=f"Confusion Matrix (Baseline {baseline_id})"
                         )


                    # Save baseline model artifact (logs path to MLflow)
                    save_path_base = self._save_model_artifact(baseline_model, "baseline", baseline_id)
                    if save_path_base:
                         # Optionally log the model itself if PyCaret didn't
                         # mlflow.log_artifact(f"{save_path_base}.pkl", artifact_path="models/baseline")
                         pass

                    mlflow.log_metric("model_duration_seconds", time.time() - start_time_model)
                    mlflow.set_tag("status", "completed")
                    success_count += 1

            except Exception as e:
                logging.error(f"Error processing baseline model '{baseline_id}': {e}", exc_info=True)
                mlflow.set_tag("status", "failed")
                mlflow.log_param("error", str(e))
                # Continue to the next baseline model

        logging.info(f"Baseline analysis finished. Successfully processed {success_count}/{len(baseline_model_ids)} models.")
        return success_count > 0 # Consider success if at least one baseline worked


    def tune_best_model(self) -> bool:
        if self.best_model_obj is None or self.pycaret_module is None:
            logging.error("Best model or PyCaret module not available for tuning.")
            return False
        if self.results_df is None or self.results_df.empty:
             logging.warning("Results dataframe is empty, cannot reliably get best model ID for logging.")
             best_model_id = "unknown_best" # Fallback ID
        else:
            # Ensure index access is safe
            try:
                best_model_id = self.results_df.index[0]
            except IndexError:
                logging.warning("Results dataframe index is out of bounds.")
                best_model_id = "unknown_best"


        logging.info(f"--- Tuning Best Model: {best_model_id} (Optimize: {self.optimize_metric}) ---")
        start_time = time.time()

        tune_params = {
            "estimator": self.best_model_obj,
            "optimize": self.optimize_metric,
            "fold": self.config.get("tuning_folds", 5),
            "n_iter": self.config.get("tuning_iterations", 10), # Default to 10 iterations if not set
            "search_library": self.config.get("tuning_search_library", 'scikit-learn'),
            "search_algorithm": self.config.get("tuning_search_algorithm", 'random'),
            # "n_jobs": self.n_jobs, # <--- REMOVE THIS LINE
            # "custom_grid": self.config.get("custom_grid", None), # Add if using grid search
            "verbose": False # Reduce console noise during tuning
        }
        # Log n_jobs from setup for clarity, as tune_model will use it internally
        logging.info(f"Tuning Params: library={tune_params['search_library']}, algo={tune_params['search_algorithm']}, iters={tune_params['n_iter']}, folds={tune_params['fold']}. Using n_jobs={self.n_jobs} (set during setup).")

        try:
            # PyCaret's tune_model will internally handle n_jobs based on setup when using optuna
            self.tuned_best_model = self.pycaret_module.tune_model(**tune_params)
            tuning_duration = time.time() - start_time
            logging.info(f"Model tuning completed in {tuning_duration:.2f} seconds.")
            logging.info("Tuned Model Pipeline:")
            print(self.tuned_best_model) # Print pipeline for inspection

            # Log tuning parameters explicitly
            mlflow.log_params({
                 "tune_optimize_metric": self.optimize_metric,
                 "tune_folds": tune_params["fold"],
                 "tune_n_iter": tune_params["n_iter"],
                 "tune_search_library": tune_params["search_library"],
                 "tune_search_algorithm": tune_params["search_algorithm"],
                 "tune_duration_seconds": tuning_duration,
            })
            # Pull tuning results grid
            tuned_results = self.pycaret_module.pull()
            logging.info("\nTuned Model Performance (Cross-Validated):")
            print(tuned_results)
            # Log the key optimization metric after tuning
            if tuned_results is not None and not tuned_results.empty and self.optimize_metric in tuned_results.columns:
                 # Ensure 'Mean' row exists
                 if 'Mean' in tuned_results.index:
                     tuned_metric_val = tuned_results.loc['Mean', self.optimize_metric]
                     mlflow.log_metric(f"tuned_cv_{self.optimize_metric}", tuned_metric_val)
                 else:
                      logging.warning(f"'Mean' row not found in tune_model results. Cannot log mean {self.optimize_metric}.")
            elif tuned_results is None or tuned_results.empty:
                 logging.warning("tune_model did not return results dataframe.")
            else: # results dataframe exists but metric is missing
                 logging.warning(f"Optimize metric '{self.optimize_metric}' not found in tune_model results columns.")


            return True
        except Exception as e:
            logging.error(f"Failed to tune model {best_model_id}: {e}", exc_info=True)
            mlflow.log_param("tuning_error", str(e))
            return False
        
    def analyze_tuned_model(self) -> bool:
        if self.tuned_best_model is None or self.pycaret_module is None:
            logging.error("Tuned model or PyCaret module not available for analysis.")
            return False

        best_model_id = self._get_model_id(self.tuned_best_model) or "tuned_best"
        logging.info(f"--- Analyzing Tuned Model: {best_model_id} ---")

        try:
            # Evaluate model (logs metrics via log_experiment, shows plots interactively)
            logging.info("Evaluating tuned model (hold-out set)...")
            # _ = self.pycaret_module.evaluate_model(self.tuned_best_model) # Interactive plots
            # Log specific plots we definitely want as artifacts
            self._log_plot_artifact(
                 plot_function=lambda save, verbose: self.pycaret_module.plot_model(self.tuned_best_model, plot='feature', save=save, verbose=verbose),
                 plot_name=f"Feature Importance ({best_model_id})"
            )
            # Add other relevant plots based on task type
            if self.task_type == 'classification':
                 self._log_plot_artifact(
                     plot_function=lambda save, verbose: self.pycaret_module.plot_model(self.tuned_best_model, plot='auc', save=save, verbose=verbose),
                     plot_name=f"AUC ({best_model_id})"
                 )
                 self._log_plot_artifact(
                     plot_function=lambda save, verbose: self.pycaret_module.plot_model(self.tuned_best_model, plot='confusion_matrix', save=save, verbose=verbose),
                     plot_name=f"Confusion Matrix ({best_model_id})"
                 )
            elif self.task_type == 'regression':
                  self._log_plot_artifact(
                     plot_function=lambda save, verbose: self.pycaret_module.plot_model(self.tuned_best_model, plot='residuals', save=save, verbose=verbose),
                     plot_name=f"Residuals Plot ({best_model_id})"
                  )
                  self._log_plot_artifact(
                     plot_function=lambda save, verbose: self.pycaret_module.plot_model(self.tuned_best_model, plot='error', save=save, verbose=verbose),
                     plot_name=f"Prediction Error Plot ({best_model_id})"
                  )


            # Conditional SHAP plot (more robust check needed for complex pipelines)
            # The check based on `best_model_id` from compare_models might be brittle if tuning drastically changed the estimator type.
            # A safer check might involve inspecting the actual estimator in the tuned pipeline.
            actual_estimator_id = self._get_model_id(self.tuned_best_model) # Try to get ID from pipeline
            supported_shap_summary_models = {'et', 'lightgbm', 'xgboost', 'rf', 'dt', 'catboost'} # Models supporting interpret_model plot='summary'

            if self.task_type == 'classification' and actual_estimator_id in supported_shap_summary_models:
                logging.info(f"Attempting SHAP summary plot for '{actual_estimator_id}'...")
                self._log_plot_artifact(
                     plot_function=lambda save, verbose: self.pycaret_module.interpret_model(self.tuned_best_model, plot='summary', save=save), # interpret doesn't use verbose
                     plot_name=f"SHAP Summary Plot ({best_model_id})"
                )
            elif self.task_type == 'classification':
                logging.warning(f"SHAP summary plot via interpret_model() may not be supported for model type '{actual_estimator_id}'. Skipping.")

            return True
        except Exception as e:
            logging.error(f"Failed during tuned model analysis: {e}", exc_info=True)
            mlflow.log_param("analysis_error", str(e))
            return False

    def _get_model_id(self, model_pipeline: Any) -> Optional[str]:
        """Tries to extract the PyCaret model ID from a pipeline object."""
        try:
            # PyCaret >= 3.x often uses 'actual_estimator'
            if hasattr(model_pipeline, 'named_steps') and 'actual_estimator' in model_pipeline.named_steps:
                 estimator = model_pipeline.named_steps['actual_estimator']
            # Older versions or different structures might have it at the end
            elif hasattr(model_pipeline, 'steps'):
                 estimator = model_pipeline.steps[-1][1]
            else: # Might be the object itself if not a pipeline
                 estimator = model_pipeline

            # Find the ID from PyCaret's internal mapping (this is a bit fragile)
            estimator_name = estimator.__class__.__name__
            all_models = self.pycaret_module.models()
            model_id = all_models[all_models['Class'] == estimator_name].index.tolist()
            if model_id:
                 return model_id[0]
            else:
                 # Fallback: try matching common IDs if direct class lookup fails
                 name_lower = estimator_name.lower()
                 if 'logistic' in name_lower: return 'lr'
                 if 'randomforest' in name_lower: return 'rf'
                 if 'lgbm' in name_lower: return 'lightgbm'
                 # Add more mappings as needed
                 logging.warning(f"Could not map estimator class '{estimator_name}' to PyCaret ID.")
                 return None
        except Exception as e:
            logging.warning(f"Could not determine model ID from pipeline: {e}")
            return None


    def finalize_and_save_model(self) -> bool:
        if self.tuned_best_model is None or self.pycaret_module is None:
            logging.error("Tuned model or PyCaret module not available for finalization.")
            return False

        best_model_id = self._get_model_id(self.tuned_best_model) or "final_tuned"
        logging.info(f"--- Finalizing and Saving Model: {best_model_id} ---")
        start_time = time.time()
        try:
            # Finalize model (train on full dataset including holdout)
            self.final_model = self.pycaret_module.finalize_model(self.tuned_best_model)
            logging.info(f"Model finalized in {time.time() - start_time:.2f} seconds. Type: {type(self.final_model)}")

            # Save final model artifact and metadata (logs path to MLflow)
            save_path_base = self._save_model_artifact(self.final_model, "final", best_model_id)
            if save_path_base:
                # Optionally log the final model artifact itself if PyCaret didn't
                # mlflow.log_artifact(f"{save_path_base}.pkl", artifact_path="models/final")
                return True
            else:
                return False # Saving failed

        except Exception as e:
            logging.error(f"Failed to finalize or save model: {e}", exc_info=True)
            mlflow.log_param("finalize_error", str(e))
            return False


    def run_pipeline(self):
        """Executes the full AutoML pipeline sequentially."""
        start_run_time = time.time()
        run_status = "FAILED" # Default status
        mlflow_run_id = None

        run_name = f"AutoML_{self.config.get('experiment_name', 'exp')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            with mlflow.start_run(run_name=run_name) as run:
                 mlflow_run_id = run.info.run_id
                 logging.info(f"Starting MLflow Run: {run_name} (ID: {mlflow_run_id})")
                 # Log config excluding potentially sensitive info if needed
                 mlflow.log_params({k: v for k, v in self.config.items() if k != 'data_file_path'}) # Example exclude
                 mlflow.log_param("data_file_path_basename", os.path.basename(self.config['data_file_path']))
                 mlflow.set_tag("pipeline_status", "started")
                 mlflow.set_tag("pycaret_version", pycaret.__version__)

                 pipeline_steps = [
                     ("load_data", self.load_data),
                     ("detect_task_type", self.detect_task_type),
                     ("setup_pycaret", self.setup_pycaret),
                     ("compare_models", self.compare_models),
                     ("analyze_baseline_models", self.analyze_baseline_models),
                     ("tune_best_model", self.tune_best_model),
                     ("analyze_tuned_model", self.analyze_tuned_model),
                     ("finalize_and_save_model", self.finalize_and_save_model),
                 ]

                 success = True
                 for step_name, step_func in pipeline_steps:
                      logging.info(f"--- Running Step: {step_name} ---")
                      step_start_time = time.time()
                      step_success = step_func()
                      step_duration = time.time() - step_start_time
                      mlflow.log_metric(f"step_{step_name}_duration_sec", step_duration)

                      if step_name == "detect_task_type" and step_success:
                           mlflow.log_param("detected_task_type", self.task_type)

                      if not step_success:
                           logging.error(f"Pipeline step '{step_name}' failed. Aborting subsequent steps.")
                           mlflow.set_tag(f"step_{step_name}_status", "failed")
                           success = False
                           break # Exit the loop on failure
                      else:
                           mlflow.set_tag(f"step_{step_name}_status", "completed")
                           logging.info(f"Step '{step_name}' completed successfully in {step_duration:.2f} seconds.")


                 if success:
                     run_status = "COMPLETED"
                     logging.info("--- AutoML Pipeline Completed Successfully ---")
                 else:
                     run_status = "FAILED"
                     logging.error("--- AutoML Pipeline Failed ---")


        except Exception as e:
             run_status = "ERROR"
             logging.error(f"--- Unhandled Exception in AutoML Pipeline: {e} ---", exc_info=True)
             if mlflow.active_run(): # Check if run context is still active
                 mlflow.set_tag("error_message", str(e))

        finally:
             end_run_time = time.time()
             duration = end_run_time - start_run_time
             logging.info(f"Pipeline finished with status: {run_status}. Total Duration: {duration:.2f} seconds.")
             if mlflow.active_run():
                 mlflow.log_metric("pipeline_duration_seconds", duration)
                 mlflow.set_tag("pipeline_status", run_status)
                 logging.info(f"MLflow Run {mlflow_run_id} status updated. Run ending.")
             # MLflow run context automatically ends here if started with 'with'


    def predict_on_new_data(self, new_data: pd.DataFrame, model_base_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Makes predictions using a saved model pipeline and its metadata.

        Args:
            new_data: DataFrame with features matching training data.
            model_base_path: Optional base path (without .pkl or _meta.json)
                             to a specific saved model. If None, loads the
                             latest 'final' model from the configured save dir.

        Returns:
            DataFrame with predictions, or None if prediction fails.
        """
        logging.info("--- Making Predictions on New Data ---")

        load_path_base = model_base_path
        pred_module = None
        task_type = None

        # --- Determine model path and load metadata ---
        try:
            if load_path_base is None:
                # Find the latest 'final' model based on timestamp in filename
                model_dir = self.config["model_save_dir"]
                final_metas = [f for f in os.listdir(model_dir) if f.startswith('final_') and f.endswith('_meta.json')]
                if not final_metas:
                    logging.error(f"No final model metadata files (*_meta.json) found in '{model_dir}'.")
                    return None

                # Extract base paths and sort by timestamp (assuming YYYYMMDD_HHMMSS format)
                # Robust timestamp extraction might be needed if format changes
                def get_timestamp_from_meta(filename):
                    parts = filename.replace('_meta.json', '').split('_')
                    if len(parts) > 1:
                        # Try to parse the last part as timestamp
                        try:
                           dt_obj = datetime.datetime.strptime(parts[-1], "%Y%m%d_%H%M%S")
                           return dt_obj
                        except ValueError:
                             # If last part isn't a timestamp, maybe second to last?
                             try:
                                 dt_obj = datetime.datetime.strptime(parts[-2], "%Y%m%d_%H%M%S")
                                 return dt_obj
                             except (ValueError, IndexError):
                                 return datetime.datetime.min # Cannot parse, put at beginning
                        except IndexError:
                            return datetime.datetime.min
                    return datetime.datetime.min # Default if no parts match

                latest_meta_file = max(final_metas, key=get_timestamp_from_meta)
                load_path_base = os.path.join(model_dir, latest_meta_file.replace('_meta.json', ''))
                logging.info(f"Identified latest final model base path: {load_path_base}")
            else:
                 logging.info(f"Using specified model base path: {load_path_base}")

            # Load metadata to get task_type
            meta_path = f"{load_path_base}_meta.json"
            pkl_path = f"{load_path_base}.pkl"

            if not os.path.exists(meta_path):
                 logging.error(f"Metadata file not found: {meta_path}")
                 return None
            if not os.path.exists(pkl_path):
                 logging.error(f"Model PKL file not found: {pkl_path}")
                 return None

            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            task_type = metadata.get("task_type")

            if task_type == 'classification':
                pred_module = pyclf
            elif task_type == 'regression':
                pred_module = pyreg
            else:
                logging.error(f"Invalid task_type '{task_type}' found in metadata file: {meta_path}")
                return None
            logging.info(f"Loaded task type '{task_type}' from metadata. Using module: {pred_module.__name__}")

        except Exception as e:
            logging.error(f"Error finding or loading model metadata: {e}", exc_info=True)
            return None

        # --- Load model and Predict ---
        try:
            loaded_pipeline = pred_module.load_model(load_path_base) # Pass base path without .pkl
            logging.info(f"Model loaded successfully from {load_path_base}.pkl")

            # Optimize new data dtypes *before* prediction if possible
            if self.config.get('optimize_pandas_dtypes', False):
                 logging.info("Optimizing dtypes of new data...")
                 new_data = optimize_dtypes(new_data.copy()) # Use copy to avoid modifying original

            predictions = pred_module.predict_model(loaded_pipeline, data=new_data)
            logging.info("Predictions generated successfully.")

            # Rename prediction columns for clarity (PyCaret 3+ convention)
            if task_type == 'classification' and hasattr(predictions, 'prediction_label'):
                 pred_label_col = self.config.get("prediction_target_column_name", 'prediction_label')
                 pred_score_col = self.config.get("prediction_score_column_name", 'prediction_score')
                 predictions = predictions.rename(columns={'prediction_label': pred_label_col, 'prediction_score': pred_score_col})
                 logging.info(f"Renamed classification output columns to '{pred_label_col}' and '{pred_score_col}'")
            elif task_type == 'regression' and hasattr(predictions, 'prediction_label'): # Regression also uses 'prediction_label' now
                 pred_label_col = self.config.get("prediction_target_column_name", 'prediction_label')
                 predictions = predictions.rename(columns={'prediction_label': pred_label_col})
                 logging.info(f"Renamed regression output column to '{pred_label_col}'")


            return predictions

        except FileNotFoundError:
             logging.error(f"Could not load model file: {load_path_base}.pkl not found (check base path).")
             return None
        except Exception as e:
             logging.error(f"An error occurred during prediction: {e}", exc_info=True)
             return None

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("<<< Starting AutoML Runner Script (Big Data Optimized) >>>")
    CONFIG_PATH = 'config.yaml'

    try:
        # 1. Load Configuration
        config = load_config(CONFIG_PATH)
        logging.info(f"PyCaret Version: {pycaret.__version__}")
        logging.info(f"MLflow Version: {mlflow.__version__}")

        # 2. Initialize and Run the Pipeline
        runner = AutoMLRunner(config=config)
        runner.run_pipeline()

        # 3. Example Prediction (Optional & Improved)
        logging.info("\n--- Running Prediction Example ---")
        # Check if the pipeline likely completed and saved a model
        model_dir = config.get("model_save_dir", "./automl_models_bigdata")
        final_models_exist = any(f.startswith('final_') and f.endswith('_meta.json') for f in os.listdir(model_dir))

        if runner.data is not None and final_models_exist:
            try:
                # Create more realistic example data (using first row is simple, but better than pure random)
                # IMPORTANT: Ensure this example data structure matches training data (excluding target)
                # A better approach in production is to have a well-defined input schema.
                example_data = runner.data.drop(columns=[runner.target_col]).head(5) # Use first 5 rows as example
                logging.info("\nExample new data for prediction (first 5 rows of original):")
                print(example_data)

                # Make prediction using the LATEST saved final model (predict_on_new_data handles finding it)
                predictions_df = runner.predict_on_new_data(new_data=example_data.copy()) # Pass a copy

                if predictions_df is not None:
                    logging.info("\nPredictions on example data:")
                    # Display relevant columns (original index + prediction)
                    pred_col_name = config.get("prediction_target_column_name", 'prediction_label')
                    if pred_col_name in predictions_df.columns:
                         print(predictions_df[[pred_col_name]]) # Show only prediction column
                    else:
                         print(predictions_df) # Show full dataframe if column name mismatch
                else:
                    logging.warning("Prediction example failed (model loading or prediction error).")

            except Exception as pred_ex:
                logging.error(f"Failed to run prediction example: {pred_ex}", exc_info=True)
        elif runner.data is None:
            logging.warning("Skipping prediction example: Training data was not loaded.")
        elif not final_models_exist:
            logging.warning(f"Skipping prediction example: No final models found in '{model_dir}'. Pipeline might have failed before saving.")

    except Exception as main_e:
         logging.critical(f"Script failed during initialization or top-level execution: {main_e}", exc_info=True)

    logging.info("<<< AutoML Runner Script Finished >>>")