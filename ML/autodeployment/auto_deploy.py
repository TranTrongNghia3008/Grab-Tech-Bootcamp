import os
import shutil
import tempfile
import warnings
import logging
import json # Needed if loading schema from .env
from pathlib import Path
from dotenv import load_dotenv # Import dotenv
# Ensure correct imports based on library structure
try:
    # Newer versions might have HfApi directly under huggingface_hub
    from huggingface_hub import HfApi, create_repo, upload_folder, logging as hf_logging
except ImportError:
    # Older versions might have it under hf_api submodule
    from huggingface_hub.hf_api import HfApi, create_repo, upload_folder
    from huggingface_hub import logging as hf_logging
    logging.warning("Imported HfApi from huggingface_hub.hf_api. Consider updating huggingface_hub.")


# --- Load environment variables from .env file ---
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Silence noisy logs from huggingface_hub if desired
# hf_logging.set_verbosity_error()


# --- PyCaret Task Mapping ---
PYCARET_MODULES = {
    'classification': 'pycaret.classification',
    'regression': 'pycaret.regression',
    # Add other tasks as needed
}

# --- Helper Function to Infer Schema (Best Effort - returns basic schema) ---
def _infer_schema_from_model(model_pipeline):
    """
    Attempts to infer feature names and basic types (numerical/other) from a loaded PyCaret pipeline.
    Returns a dictionary like: {"feature_name": {"type": "numerical"}, ...}
    This is a basic inference; providing a manual schema is recommended for complex types.
    """
    schema = {}
    try:
        feature_names = None
        # Try different attributes where feature names might be stored
        if hasattr(model_pipeline, 'feature_names_in_'):
             feature_names = model_pipeline.feature_names_in_
        elif hasattr(model_pipeline, '_feature_names_in'): # Sometimes private attribute
             feature_names = model_pipeline._feature_names_in
        elif hasattr(model_pipeline.named_steps.get('trained_model'), 'feature_names_in_'):
             feature_names = model_pipeline.named_steps['trained_model'].feature_names_in_
        elif hasattr(model_pipeline.named_steps.get('actual_estimator'), 'feature_names_in_'):
             feature_names = model_pipeline.named_steps['actual_estimator'].feature_names_in_
        else:
             # Look for preprocessing steps that might have original names
             for name, step in model_pipeline.named_steps.items():
                 if hasattr(step, 'feature_names_in_'):
                     feature_names = step.feature_names_in_
                     logging.info(f"Inferred feature names from step '{name}'")
                     break # Take the first one found in preprocessing
             if not feature_names:
                 logging.warning("Could not reliably determine feature names from pipeline steps.")
                 return None # Cannot proceed without feature names

        # Basic type inference (defaulting to numerical)
        for name in feature_names:
            # Defaulting to numerical, user should override with manual schema for categorical etc.
             schema[name] = {"type": "numerical"}
        logging.info(f"Inferred basic schema (defaulting to numerical): {schema}")
        logging.warning("Schema inference is basic. For categorical features, provide a manual schema.")
        return schema

    except Exception as e:
        logging.warning(f"Could not infer schema automatically: {e}", exc_info=True)
        return None

# --- Function to Generate Streamlit App Code (FIXED Indentation) ---
def _generate_streamlit_code(pycaret_task_module: str, schema: dict, model_filename: str, app_title: str) -> str:
    """Generates the Python code for the Streamlit application, supporting detailed schema."""
    widget_code = []
    input_vars = []

    # Ensure schema has the expected structure, default if needed
    processed_schema = {}
    for feature, details in schema.items():
        if isinstance(details, dict) and "type" in details:
            processed_schema[feature] = details
        else:
            # Fallback for basic schema format (treat as numerical/text)
            processed_schema[feature] = {"type": "numerical"} # Default assumption
            logging.warning(f"Basic schema format detected for '{feature}'. Assuming numerical. Provide detailed schema for better UI.")


    for feature, details in processed_schema.items():
        feature_type = details.get("type", "text").lower() # Default to text if type missing
        feature_values = details.get("values") # Will be None if not provided

        # Create python-safe variable names from feature names
        safe_feature_name = ''.join(c if c.isalnum() else '_' for c in feature)
        var_name = f"input_{safe_feature_name}"
        input_vars.append(var_name)

        # Generate appropriate widget based on type and values
        if feature_type == "categorical" and feature_values and isinstance(feature_values, list) and len(feature_values) > 0:
             # Ensure values are strings for selectbox options if needed
             options_list_str = [f"'{str(v)}'" for v in feature_values] # Quote values
             widget_code.append(f"{var_name} = st.selectbox(label='{feature}', options=[{', '.join(options_list_str)}], key='{var_name}')")
        elif feature_type == "numerical":
            widget_code.append(f"{var_name} = st.number_input(label='{feature}', format='%f', key='{var_name}')")
        else: # Default to text input for 'text' or unsupported/missing types/values
            if feature_type == "categorical":
                 logging.warning(f"Categorical feature '{feature}' has no values provided in schema. Falling back to text input.")
            widget_code.append(f"{var_name} = st.text_input(label='{feature}', key='{var_name}')")

    # *** FIXED INDENTATION HERE: Use 8 spaces for join ***
    widget_code_str = "\n        ".join(widget_code) # Use 8 spaces indentation
    # Use the original feature names (keys of schema) for the DataFrame columns
    input_dict_str = ", ".join([f"'{name}': {var}" for name, var in zip(processed_schema.keys(), input_vars)])

    # Use double curly braces {{ }} to escape braces intended for the generated code
    app_code = f"""
import streamlit as st
import pandas as pd
# Make sure to import the correct module dynamically based on the task
from {pycaret_task_module} import load_model, predict_model
import os
import warnings # Added to potentially suppress warnings
import logging # Added for better debugging in the Space

# --- Page Configuration (MUST BE FIRST STREAMLIT COMMAND) ---
APP_TITLE = "{app_title}"
st.set_page_config(page_title=APP_TITLE, layout="centered", initial_sidebar_state="collapsed")

# Configure simple logging for the Streamlit app
# Use Streamlit logger if available, otherwise basic config
try:
    # Attempt to get logger specific to Streamlit context
    logger = st.logger.get_logger(__name__)
except AttributeError: # Fallback for older Streamlit versions or different contexts
    # Basic logging setup if Streamlit logger isn't available
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - StreamlitApp - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


# --- Model Configuration ---
MODEL_FILE = "{model_filename}" # Relative path within the Space

# --- Processed Schema (for type checking later) ---
# Use double braces to embed the schema dict correctly in the generated code
APP_SCHEMA = {processed_schema}


# --- Load Model ---
# Use cache_resource for efficient loading
@st.cache_resource
def get_model():
    logger.info(f"Attempting to load model from file: {{MODEL_FILE}}")
    # Define the path expected by PyCaret's load_model (without extension)
    model_load_path = MODEL_FILE.replace('.pkl','')
    logger.info(f"Calculated PyCaret load path: '{{model_load_path}}'") # Escaped braces

    if not os.path.exists(MODEL_FILE):
        st.error(f"Model file '{{MODEL_FILE}}' not found in the Space repository.")
        logger.error(f"Model file '{{MODEL_FILE}}' not found at expected path.")
        return None
    try:
        # Suppress specific warnings during loading if needed
        # warnings.filterwarnings("ignore", category=UserWarning, message=".*Trying to unpickle estimator.*")
        logger.info(f"Calling PyCaret's load_model('{{model_load_path}}')...") # Escaped braces
        # Ensure PyCaret logging doesn't interfere excessively if needed
        # from pycaret.utils.generic import enable_colab
        # enable_colab() # May help manage output/logging in some environments
        model = load_model(model_load_path)
        logger.info("PyCaret's load_model executed successfully.")
        return model
    except FileNotFoundError:
        # Specific handling if load_model itself can't find related files (like preprocess.pkl)
        st.error(f"Error loading model components for '{{model_load_path}}'. PyCaret's load_model failed, possibly missing auxiliary files.") # Escaped braces
        logger.exception(f"PyCaret load_model failed for '{{model_load_path}}', likely due to missing components:") # Escaped braces
        return None
    except Exception as e:
        # Catch other potential errors during model loading
        st.error(f"An unexpected error occurred loading model '{{model_load_path}}': {{e}}") # Escaped braces around model_load_path
        logger.exception("Unexpected model loading error details:") # Log full traceback
        return None

# --- Load the model ---
model = get_model()

# --- App Layout ---
st.title(APP_TITLE) # Title now comes after page config

if model is None:
    st.error("Model could not be loaded. Please check the application logs in the Space settings for more details. Application cannot proceed.")
else:
    st.success("Model loaded successfully!") # Indicate success
    st.markdown("Provide the input features below to generate a prediction using the deployed model.")

    # --- Input Section ---
    st.header("Model Inputs")
    with st.form("prediction_form"):
        # Dynamically generated widgets based on schema (now with correct indentation)
        {widget_code_str}
        submitted = st.form_submit_button("📊 Get Prediction")

    # --- Prediction Logic & Output Section ---
    if submitted:
        st.header("Prediction Output")
        try:
            # Create DataFrame from inputs using original feature names as keys
            # The values are automatically fetched by Streamlit using the keys assigned to widgets
            input_data_dict = {{{input_dict_str}}} # Use triple braces for dict literal inside f-string
            logger.info(f"Raw input data from form: {{input_data_dict}}")
            input_data = pd.DataFrame([input_data_dict])

            # Ensure correct dtypes based on schema before prediction
            logger.info("Applying dtypes based on schema...")
            # Use APP_SCHEMA defined earlier
            for feature, details in APP_SCHEMA.items():
                 feature_type = details.get("type", "text").lower()
                 if feature in input_data.columns: # Check if feature exists
                     try:
                         current_value = input_data[feature].iloc[0]
                         # Skip conversion if value is already None or NaN equivalent
                         if pd.isna(current_value):
                              continue

                         if feature_type == 'numerical':
                             # Convert to numeric, coercing errors (users might enter text)
                             input_data[feature] = pd.to_numeric(input_data[feature], errors='coerce')
                         elif feature_type == 'categorical':
                             # Ensure categorical inputs are treated as strings by the model if needed
                             # PyCaret often expects object/string type for categoricals in predict_model
                             input_data[feature] = input_data[feature].astype(str)
                         # Add elif for other types if needed (e.g., datetime)
                         # else: # text
                         #     input_data[feature] = input_data[feature].astype(str) # Ensure string type

                     except Exception as type_e:
                         logger.warning(f"Could not convert feature '{{feature}}' (value: {{current_value}}) to type '{{feature_type}}'. Error: {{type_e}}")
                         # Decide how to handle type conversion errors, e.g., set to NaN or keep original
                         input_data[feature] = pd.NA # Set to missing if conversion fails
                 else:
                     logger.warning(f"Feature '{{feature}}' from schema not found in input form data.")


            # Handle potential NaN values from coercion or failed conversion
            if input_data.isnull().values.any():
                 st.warning("Some inputs might be invalid or missing. Attempting to handle missing values (e.g., replacing with 0 for numerical). Check logs for details.")
                 logger.warning(f"NaN values found in input data after type conversion/validation. Filling numerical with 0. Data before fill:\\n{{input_data}}")
                 # More robust imputation might be needed depending on the model
                 # Fill only numerical NaNs with 0, leave others? Or use mode for categoricals?
                 for feature, details in APP_SCHEMA.items():
                     # Check if column exists before attempting to fill
                     if feature in input_data.columns and details.get("type") == "numerical" and input_data[feature].isnull().any():
                         input_data[feature].fillna(0, inplace=True)
                 # input_data.fillna(0, inplace=True) # Previous simpler strategy
                 logger.info(f"Data after filling NaN:\\n{{input_data}}")


            st.markdown("##### Input Data Sent to Model (after processing):")
            st.dataframe(input_data)

            # Make prediction
            logger.info("Calling predict_model...")
            with st.spinner("Predicting..."):
                # Suppress prediction warnings if needed
                # with warnings.catch_warnings():
                #    warnings.simplefilter("ignore")
                predictions = predict_model(model, data=input_data)
                logger.info("Prediction successful.")

            st.markdown("##### Prediction Result:")
            logger.info(f"Prediction output columns: {{predictions.columns.tolist()}}")

            # Display relevant prediction columns (adjust based on PyCaret task)
            # Common columns: 'prediction_label', 'prediction_score'
            pred_col_label = 'prediction_label'
            pred_col_score = 'prediction_score'

            if pred_col_label in predictions.columns:
                st.success(f"Predicted Label: **{{predictions[pred_col_label].iloc[0]}}**")
            # Also show score if available for classification
            if pred_col_score in predictions.columns and pycaret_task_module == 'pycaret.classification':
                 st.info(f"Prediction Score: **{{predictions[pred_col_score].iloc[0]:.4f}}**")
            # Handle regression output (usually just score)
            elif pred_col_score in predictions.columns and pycaret_task_module == 'pycaret.regression':
                 st.success(f"Predicted Value: **{{predictions[pred_col_score].iloc[0]:.4f}}**")
            else:
                 # Fallback: Display the last column as prediction if specific ones aren't found
                 try:
                     # Exclude input columns if they are present in the output df
                     output_columns = [col for col in predictions.columns if col not in input_data.columns]
                     if output_columns:
                         last_col_name = output_columns[-1]
                         st.info(f"Prediction Output (Column: '{{last_col_name}}'): **{{predictions[last_col_name].iloc[0]}}**")
                         logger.warning(f"Could not find standard prediction columns. Displaying last non-input column: '{{last_col_name}}'")
                     else: # If only input columns are returned (unlikely)
                         st.warning("Prediction output seems to only contain input columns.")
                 except IndexError:
                     st.error("Prediction result DataFrame is empty or has unexpected format.")
                     logger.error("Prediction result DataFrame is empty or has unexpected format.")


            # Show full prediction output optionally
            with st.expander("View Full Prediction Output DataFrame"):
                st.dataframe(predictions)

        except Exception as e:
            st.error(f"An error occurred during prediction: {{e}}")
            logger.exception("Prediction error details:") # Log full traceback

"""
    return app_code


# --- Function to Generate requirements.txt ---
def _generate_requirements(pycaret_version=None):
    """Generates the content for requirements.txt."""
    if pycaret_version:
        # Ensure pycaret version format is correct (e.g., 3.1.0, not just 3.1)
        pycaret_dep = f"pycaret[full]=={pycaret_version}"
    else:
        pycaret_dep = "pycaret[full]" # Using [full] increases Space size but ensures most dependencies
        logging.warning("PyCaret version not specified in config. Using 'pycaret[full]'. Pinning the version is recommended for reproducibility.")

    requirements = f"""
streamlit>=1.20.0 # Specify a recent streamlit version
{pycaret_dep}
pandas>=1.5.0 # Specify recent pandas
scikit-learn # Often needed by PyCaret under the hood, ensure compatibility with pycaret version if possible
python-dotenv # Needed if app uses .env itself, though typically not needed in deployed Space
# Add any other specific dependencies your model might need here
# Example: numpy, joblib, etc.
"""
    return requirements.strip()


# --- Function to Generate README.md with SDK info ---
def _generate_readme(app_title: str) -> str:
    """Generates basic README content specifying the SDK."""
    readme_content = f"""
---
title: {app_title}
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---

# {app_title}

Simple prediction app deployed using Streamlit. Model file: `model.pkl`.
Check `requirements.txt` for dependencies.
"""
    return readme_content.strip()


# --- Main Deployment Function (Accepts detailed schema) ---
def deploy_pycaret_streamlit_app(
    model_path: str,          # Specific model to deploy
    pycaret_task: str,        # Task for this model
    app_name: str,            # Name for this specific deployment
    pycaret_version: str = None, # Optional override for PyCaret version
    feature_schema: dict = None, # Optional override for schema (can be detailed)
    hf_space_id: str = None      # Optional override for space ID
) -> str | None:
    """
    Generates a Streamlit app for a PyCaret model and deploys it to Hugging Face Spaces.

    Args:
        model_path: Path to the saved PyCaret model (.pkl file).
        pycaret_task: The task type (e.g., 'classification', 'regression').
        app_name: Desired name for the Hugging Face Space (and app title). Should be URL-friendly.
        pycaret_version: Specific version of PyCaret used for training (e.g., "3.1.0"). Overrides .env default.
        feature_schema: Optional dict defining input features.
                        If None, attempts basic inference.
                        For detailed UI, provide:
                        {
                            "feature_name_1": {"type": "numerical"},
                            "feature_name_2": {"type": "categorical", "values": ["A", "B", "C"]},
                            "feature_name_3": {"type": "text"}
                        }
                        Supported types: "numerical", "categorical", "text".
                        "categorical" requires a "values" list for selectbox.
        hf_space_id: Optional explicit Hugging Face Space ID (e.g., "my-user/my-cool-app").
                     If None, it's constructed from HF_USERNAME and app_name.

    Returns:
        The URL of the deployed Hugging Face Space, or None if deployment fails.
    """
    logging.info(f"Starting deployment process for model: {model_path}")

    # --- Get Credentials and User Info from Environment ---
    hf_username = os.getenv("HF_USERNAME")
    hf_token = os.getenv("HF_TOKEN")

    if not hf_username:
        logging.error("HF_USERNAME not found in environment variables or .env file.")
        return None
    if not hf_token:
        logging.error("HF_TOKEN not found in environment variables or .env file.")
        return None

    # --- Input Validation ---
    model_file = Path(model_path)
    if not model_file.is_file() or model_file.suffix != '.pkl':
        logging.error(f"Invalid model path: {model_path}. Must be a .pkl file.")
        return None

    if pycaret_task not in PYCARET_MODULES:
        logging.error(f"Unsupported PyCaret task: {pycaret_task}. Supported: {list(PYCARET_MODULES.keys())}")
        return None

    # Validate app_name for URL safety (basic check)
    if not all(c.isalnum() or c in '-_' for c in app_name):
        logging.warning(f"App name '{app_name}' contains characters potentially unsafe for URLs. Consider using only letters, numbers, hyphens, or underscores.")


    # --- Determine PyCaret Module ---
    pycaret_module = None # Initialize to None
    module_name = None
    try:
        module_name = PYCARET_MODULES[pycaret_task]
        # Use importlib for cleaner dynamic imports if possible, fallback to exec
        try:
            import importlib
            pycaret_module = importlib.import_module(module_name)
        except ImportError:
            logging.warning(f"Could not import {module_name} using importlib, falling back to exec.")
            exec(f"import {module_name} as pycaret_module_exec", globals())
            pycaret_module = globals()['pycaret_module_exec']

        logging.info(f"Successfully imported PyCaret module: {module_name}")
    except ImportError:
        logging.error(f"Failed to import PyCaret module: {module_name}. Is PyCaret ({pycaret_version or 'latest'}) installed in the environment running this script?")
        return None
    except KeyError:
         logging.error(f"Invalid PyCaret task key: {pycaret_task}")
         return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during PyCaret module import: {e}", exc_info=True)
        return None


    # --- Load Model Locally for Schema Inference (if needed) ---
    final_schema = feature_schema # Use provided schema if available
    if not final_schema:
        logging.info("No manual schema provided. Attempting to infer basic schema from the model...")
        if pycaret_module is None:
             logging.error("PyCaret module was not loaded correctly. Cannot infer schema.")
             return None
        try:
            logging.info(f"Loading model {model_file.with_suffix('')} locally for schema inference...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Ensure the function exists before calling
                if hasattr(pycaret_module, 'load_model'):
                    temp_model = pycaret_module.load_model(str(model_file.with_suffix('')))
                else:
                    logging.error(f"'load_model' function not found in module {pycaret_module.__name__}")
                    # Clean up potentially loaded module via exec
                    if 'pycaret_module_exec' in globals(): del globals()['pycaret_module_exec']
                    return None
            logging.info("Local model loaded for inference.")
            final_schema = _infer_schema_from_model(temp_model) # Assign inferred schema
            del temp_model # Free up memory
            if not final_schema:
                 logging.error("Failed to infer schema. Cannot generate UI.")
                 if 'pycaret_module_exec' in globals(): del globals()['pycaret_module_exec']
                 return None
        except Exception as e:
            logging.error(f"Error loading model locally for schema inference: {e}", exc_info=True)
            if 'pycaret_module_exec' in globals(): del globals()['pycaret_module_exec']
            return None

    if not final_schema: # Double check if schema is missing
        logging.error("Feature schema is required but could not be determined or provided.")
        if 'pycaret_module_exec' in globals(): del globals()['pycaret_module_exec']
        return None
    logging.info(f"Using final feature schema for UI generation: {final_schema}")


    # --- Determine PyCaret Version (Argument > .env > None) ---
    final_pycaret_version = pycaret_version if pycaret_version is not None else os.getenv("DEFAULT_PYCARET_VERSION") or None
    logging.info(f"Target PyCaret version for requirements.txt: {final_pycaret_version or 'Not specified (using [full])'}")


    # --- Generate Files ---
    logging.info("Generating Streamlit app code, requirements file, and README.")
    deployed_model_filename = "model.pkl" # Standard name inside the Space
    try:
        # Ensure module_name is defined before calling _generate_streamlit_code
        if module_name is None:
             logging.error("PyCaret module name could not be determined. Cannot generate Streamlit code.")
             if 'pycaret_module_exec' in globals(): del globals()['pycaret_module_exec']
             return None
        # Pass the final schema (either inferred or provided)
        streamlit_code = _generate_streamlit_code(module_name, final_schema, deployed_model_filename, app_name)
        requirements_content = _generate_requirements(final_pycaret_version)
        readme_content = _generate_readme(app_name) # Generate README content
    except Exception as e:
        logging.error(f"Error generating app code or requirements: {e}", exc_info=True)
        if 'pycaret_module_exec' in globals(): del globals()['pycaret_module_exec']
        return None


    # --- Package and Deploy ---
    repo_id = hf_space_id if hf_space_id else f"{hf_username}/{app_name}"
    logging.info(f"Preparing deployment to Hugging Face Space: {repo_id}")

    try:
        # Create a temporary directory for packaging
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Save generated files
            with open(tmpdir_path / "app.py", "w", encoding="utf-8") as f: f.write(streamlit_code)
            with open(tmpdir_path / "requirements.txt", "w", encoding="utf-8") as f: f.write(requirements_content)
            with open(tmpdir_path / "README.md", "w", encoding="utf-8") as f: f.write(readme_content) # Save README.md

            # Copy the model file with the standard name
            shutil.copyfile(model_file, tmpdir_path / deployed_model_filename)
            logging.info(f"Files packaged in temporary directory: {tmpdir}")

            # Use HfApi for robust interaction
            api = HfApi(token=hf_token)

            # Create Space repository on Hugging Face Hub
            # Set space_sdk="docker" as per the previous error message
            # The README.md file will tell HF to use the streamlit runner
            create_repo(
                repo_id=repo_id,
                token=hf_token,
                repo_type="space",
                space_sdk="docker", # Use "docker" as the base SDK type
                exist_ok=True
            )
            logging.info(f"Ensured Space repository exists or is created with base SDK 'docker': {repo_id}")

            # Upload folder contents to the Space
            logging.info("Uploading files (including README.md specifying streamlit SDK) to the Space repository...")
            commit_info = api.upload_folder(
                folder_path=str(tmpdir_path), # Ensure folder_path is a string
                repo_id=repo_id,
                repo_type="space",
                commit_message=f"Deploy PyCaret model {model_file.name} with fixed indentation", # Updated commit message
            )

            logging.info(f"Application files uploaded successfully. Commit: {commit_info.commit_url}")

            # Construct the Space URL
            space_url = f"https://huggingface.co/spaces/{repo_id}"
            logging.info(f"Deployment initiated. Your app should be available shortly at: {space_url}")
            print(f"\n🚀 Streamlit App Deployed! URL: {space_url}\n")
            return space_url

    except Exception as e:
        logging.error(f"An error occurred during Hugging Face API interaction or deployment: {e}", exc_info=True)
        print(f"\n❌ Deployment Failed. Error: {e}\n")
        return None
    finally:
         # Clean up dynamically imported module (exec fallback)
         if 'pycaret_module_exec' in globals():
             del globals()['pycaret_module_exec']
             logging.debug("Cleaned up dynamically imported pycaret module (exec).")


# --- Example Usage (Demonstrating detailed schema) ---
if __name__ == "__main__":
    print("Running deployment example using .env configuration...")

    # --- Load Configuration from .env or use fallbacks ---
    hf_username_check = os.getenv("HF_USERNAME")
    hf_token_check = os.getenv("HF_TOKEN")

    if not hf_username_check or not hf_token_check:
        print("\n❌ Error: HF_USERNAME and HF_TOKEN must be set in your .env file or environment variables.")
        exit(1)

    model_to_deploy = os.getenv("DEFAULT_MODEL_PATH", "YOUR_MODEL.pkl")
    task_for_model = os.getenv("DEFAULT_PYCARET_TASK", "classification")
    name_for_app = os.getenv("DEFAULT_APP_NAME", "my-pycaret-app")
    version_for_pycaret = os.getenv("DEFAULT_PYCARET_VERSION")

    # --- !!! OPTIONAL: Define Detailed Schema Here !!! ---
    # If you know your features, define them here for a better UI.
    # Otherwise, set to None to use basic inference.
    manual_schema = None
    # Example of a detailed schema:
    # manual_schema = {
    #     "age": {"type": "numerical"},
    #     "sex": {"type": "categorical", "values": ["Male", "Female"]},
    #     "bmi": {"type": "numerical"},
    #     "children": {"type": "numerical"}, # Or categorical if few discrete values
    #     "smoker": {"type": "categorical", "values": ["yes", "no"]},
    #     "region": {"type": "categorical", "values": ["southwest", "southeast", "northwest", "northeast"]}
    # }
    # --- /!!! OPTIONAL: Define Detailed Schema Here !!! ---


    # --- !!! VERIFY SETTINGS BEFORE RUNNING !!! ---
    print("-" * 30)
    print("Deployment Configuration:")
    print(f"  Model Path:       {model_to_deploy}")
    print(f"  PyCaret Task:     {task_for_model}")
    print(f"  App Name:         {name_for_app}")
    print(f"  PyCaret Version:  {version_for_pycaret or 'Not Specified (using latest)'}")
    print(f"  Hugging Face User:{hf_username_check}")
    print(f"  Manual Schema:    {'Provided' if manual_schema else 'No (will attempt basic inference)'}")
    print("-" * 30)
    if manual_schema:
        print("Using provided manual schema:")
        print(json.dumps(manual_schema, indent=2))
        print("-" * 30)


    # --- Basic Check ---
    model_path_obj = Path(model_to_deploy)
    if model_to_deploy == "YOUR_MODEL.pkl" or not model_path_obj.exists():
         print(f"\n⚠️ Warning: Model path '{model_to_deploy}' seems incorrect or file doesn't exist.")
         print("Please update DEFAULT_MODEL_PATH in your .env file or ensure the path is correct.")
         # Decide whether to exit or proceed
         # exit(1) # Uncomment to stop execution if model path is invalid

    # --- Proceed with Deployment ---
    if model_path_obj.exists(): # Only proceed if model exists
        deployed_url = deploy_pycaret_streamlit_app(
            model_path=str(model_path_obj), # Pass as string
            pycaret_task=task_for_model,
            app_name=name_for_app,
            pycaret_version=version_for_pycaret,
            feature_schema=manual_schema # Pass the schema here
        )

        if deployed_url:
            print(f"Access your deployed application at: {deployed_url}")
        else:
            print("Deployment failed. Check logs above for details.")
    else:
        print(f"Deployment skipped because model file was not found at: {model_to_deploy}")

