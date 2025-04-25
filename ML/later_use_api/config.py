# config.py
import os
from dotenv import load_dotenv

load_dotenv() # Load variables from a .env file if it exists

# --- Configuration ---
# Default values, can be overridden by API requests or environment variables
DEFAULT_UNIQUE_VALUE_THRESHOLD_FOR_CLASSIFICATION = 20
DEFAULT_SESSION_ID = 123
DEFAULT_BASELINE_CLASSIFICATION_MODEL = 'dt'
DEFAULT_BASELINE_REGRESSION_MODEL = 'lr'
DEFAULT_EXPERIMENT_NAME_PREFIX = 'fastapi_automl_exp'
DEFAULT_SAVE_MODEL_DIR = os.getenv("MODEL_SAVE_DIR", "saved_models") # Store models here
DEFAULT_PLOT_SAVE_DIR = os.getenv("PLOT_SAVE_DIR", "saved_plots") # Store plots here
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", None) # Optional MLflow integration

# Create directories if they don't exist
os.makedirs(DEFAULT_SAVE_MODEL_DIR, exist_ok=True)
os.makedirs(DEFAULT_PLOT_SAVE_DIR, exist_ok=True)

# Define metrics based on task type (could be more sophisticated)
CLASSIFICATION_SORT_METRIC = 'Accuracy'
CLASSIFICATION_OPTIMIZE_METRIC = 'Accuracy'
REGRESSION_SORT_METRIC = 'R2'
REGRESSION_OPTIMIZE_METRIC = 'R2'