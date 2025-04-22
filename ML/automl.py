import pandas as pd
# Import specific modules with aliases to avoid name conflicts
import pycaret.classification as pyclf
import pycaret.regression as pyreg
import time
import numpy as np # For checking numeric types

# --- Configuration ---
DATA_FILE_PATH = 'E:/Grab Bootcamp/Grab-Tech-Bootcamp/ML/train.csv'
TARGET_COLUMN = 'Survived'
# Define a threshold for unique values to distinguish classification from regression
# Adjust this based on your typical datasets
UNIQUE_VALUE_THRESHOLD_FOR_CLASSIFICATION = 20 
SESSION_ID = 123
BASELINE_CLASSIFICATION_MODEL = 'dt' # e.g., 'lr' (Logistic), 'dt' (Decision Tree)
BASELINE_REGRESSION_MODEL = 'lr'   # e.g., 'lr' (Linear), 'dt' (Decision Tree), 'ridge'
EXPERIMENT_NAME = 'automl_autodetect_exp'
SAVE_MODEL_NAME = 'final_automl_pipeline_autodetect'

# --- 1. Load Data ---
try:
    data = pd.read_csv(DATA_FILE_PATH)
    print(f"Data loaded successfully from '{DATA_FILE_PATH}' with shape: {data.shape}")
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the dataset.")
    data_subset = data # Use the full data for now
except FileNotFoundError:
    print(f"Error: Data file not found at '{DATA_FILE_PATH}'")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 2. Auto-Detect Task Type ---
print("\n--- Auto-Detecting Task Type ---")
target_series = data_subset[TARGET_COLUMN]
n_unique = target_series.nunique()
dtype = target_series.dtype

task_type = None

if pd.api.types.is_object_dtype(dtype):
    task_type = 'classification'
    print(f"Detected task type: Classification (Target dtype is object)")
elif pd.api.types.is_numeric_dtype(dtype):
    # Check for binary classification explicitly (only 0s and 1s)
    unique_values = target_series.unique()
    if len(unique_values) == 2 and np.all(np.isin(unique_values, [0, 1])):
         task_type = 'classification'
         print(f"Detected task type: Binary Classification (Target has values {unique_values})")
    elif n_unique < UNIQUE_VALUE_THRESHOLD_FOR_CLASSIFICATION:
        task_type = 'classification'
        print(f"Detected task type: Classification (Target dtype is {dtype}, {n_unique} unique values < threshold {UNIQUE_VALUE_THRESHOLD_FOR_CLASSIFICATION})")
        # Optional: Force target to integer or category if it's float but looks categorical
        # if pd.api.types.is_float_dtype(dtype):
        #    print("Warning: Target is float but has few unique values. Consider converting to int/category if appropriate.")
    else:
        task_type = 'regression'
        print(f"Detected task type: Regression (Target dtype is {dtype}, {n_unique} unique values >= threshold {UNIQUE_VALUE_THRESHOLD_FOR_CLASSIFICATION})")
else:
    print(f"Warning: Could not automatically determine task type for target column '{TARGET_COLUMN}' with dtype {dtype}. Defaulting to classification, but please verify.")
    task_type = 'classification' # Default fallback, might need adjustment

# --- 3. Setup PyCaret Environment ---
print("\nSetting up PyCaret environment...")
start_time = time.time()

setup_env = None
pycaret_module = None
sort_metric = None
optimize_metric = None
baseline_model_name_default = None

if task_type == 'classification':
    pycaret_module = pyclf
    sort_metric = 'Accuracy' # Common default, could use 'AUC', 'F1' etc.
    optimize_metric = 'Accuracy'
    baseline_model_name_default = BASELINE_CLASSIFICATION_MODEL
    print(f"Using PyCaret Classification module. Sorting/Optimizing by: {sort_metric}")
    setup_env = pycaret_module.setup(
        data=data_subset,
        target=TARGET_COLUMN,
        session_id=SESSION_ID,
        log_experiment=False, # Avoids MLflow logging for simplicity
        experiment_name=EXPERIMENT_NAME + '_clf',
        # Add any specific classification setup args here if needed
        numeric_imputation = 'mean',
        categorical_imputation = 'mode',
    )

elif task_type == 'regression':
    pycaret_module = pyreg
    sort_metric = 'R2' # Common default, could use 'RMSE', 'MAE' etc.
    optimize_metric = 'R2'
    baseline_model_name_default = BASELINE_REGRESSION_MODEL
    print(f"Using PyCaret Regression module. Sorting/Optimizing by: {sort_metric}")
    setup_env = pycaret_module.setup(
        data=data_subset,
        target=TARGET_COLUMN,
        session_id=SESSION_ID,
        log_experiment=False, # Avoids MLflow logging for simplicity
        experiment_name=EXPERIMENT_NAME + '_reg',
        # Add any specific regression setup args here if needed
        numeric_imputation = 'mean',
        normalize = True,
    )
else:
    print("Error: Task type could not be determined. Exiting.")
    exit()

print(f"PyCaret setup completed in {time.time() - start_time:.2f} seconds.")

# --- 4. Compare Models ---
print("\nComparing models...")
start_time = time.time()

# This trains and evaluates common models using cross-validation
# Pass the appropriate module and sort metric
best_model_obj = pycaret_module.compare_models(sort=sort_metric)

# Get the results grid as a DataFrame
results_df = pycaret_module.pull()
print("\nModel Comparison Results (Cross-Validated):")
print(results_df)
print(f"Model comparison completed in {time.time() - start_time:.2f} seconds.")

# --- 5. Baseline Model Analysis ---
print("\n--- Baseline Model Analysis ---")
# Choose a simple, interpretable model based on task type.
baseline_model_name = baseline_model_name_default

# Check if the chosen default baseline exists in the results
if baseline_model_name not in results_df.index:
    print(f"Warning: Default baseline '{baseline_model_name}' not available from compare_models.")
    # Fallback: Try other simple models or just pick the first one
    simple_models = ['lr', 'ridge', 'lasso', 'dt', 'dummy'] # Common simple ones
    available_simple = [m for m in simple_models if m in results_df.index]
    if available_simple:
        baseline_model_name = available_simple[0]
        print(f"Using fallback baseline: {baseline_model_name}")
    else: # If no simple models ran (unlikely), use the top model as baseline ref
        baseline_model_name = results_df.index[0]
        print(f"Using top model as baseline reference: {baseline_model_name}")

print(f"Selected baseline model: {baseline_model_name}")

# Create the specific baseline model instance (trained on CV folds)
try:
    baseline_model = pycaret_module.create_model(baseline_model_name)

    print("\nEvaluating Baseline Model:")
    # Evaluate on the hold-out set
    pycaret_module.evaluate_model(baseline_model)

    # Get specific plots, including feature importance
    try:
        # For Decision Trees: plot tree
        if baseline_model_name == 'dt':
            pycaret_module.plot_model(baseline_model, plot = 'tree', save=True)
            print(f"Baseline decision tree plot saved.") # Name is auto-generated

        # Plot feature importance (works for most models)
        plot_type = 'feature'
        # Linear models in regression might use 'coef' instead of 'feature' - adjust if needed
        # if task_type == 'regression' and baseline_model_name in ['lr', 'ridge', 'lasso']:
        #     plot_type = 'coef'

        pycaret_module.plot_model(baseline_model, plot=plot_type, save=True)
        print(f"Baseline feature importance plot saved.") # Name is auto-generated

    except Exception as e:
        print(f"Could not generate plots for baseline model {baseline_model_name}: {e}")

except Exception as e:
     print(f"Error creating or evaluating baseline model '{baseline_model_name}': {e}")


# --- 6. Complex Model Tuning & Analysis ---
print("\n--- Complex Model Analysis (Best Performing Model) ---")
print(f"Best overall model identified by compare_models: {best_model_obj.__class__.__name__}")

# Tune the hyperparameters of the *best* model found by compare_models
print("\nTuning the best model...")
start_time = time.time()

# Use the optimize_metric determined earlier
tuned_best_model = pycaret_module.tune_model(best_model_obj, optimize=optimize_metric)
print(f"\nModel tuning completed in {time.time() - start_time:.2f} seconds.")
print("Tuned Model Parameters:")
print(tuned_best_model) # Shows the model pipeline with tuned hyperparameters

print("\nEvaluating Tuned Complex Model:")
# Evaluate the *tuned* model on the hold-out set
pycaret_module.evaluate_model(tuned_best_model)

# Plot feature importance for the complex model
try:
    pycaret_module.plot_model(tuned_best_model, plot='feature', save=True)
    print(f"Tuned model feature importance plot saved.")
except Exception as e:
    print(f"Could not generate feature importance plot for tuned model: {e}")

# --- 7. Finalize and Save Model ---
print("\nFinalizing model (training on full dataset)...")
# Retrain the tuned model pipeline on the entire dataset
final_model = pycaret_module.finalize_model(tuned_best_model)

print(f"\nSaving the final model pipeline as '{SAVE_MODEL_NAME}.pkl'...")
pycaret_module.save_model(final_model, SAVE_MODEL_NAME) # Saves as .pkl file

print("\n--- AutoML Process Complete ---")
print(f"Task type detected: {task_type}")
print(f"Baseline model analyzed: {baseline_model_name}")
print(f"Best performing model tuned: {final_model.__class__.__name__}")
print(f"Final pipeline saved as '{SAVE_MODEL_NAME}.pkl'")

# --- 8. Making Predictions (Example) ---
print("\n--- Prediction Example (using saved model) ---")
try:
    loaded_pipeline = pycaret_module.load_model(SAVE_MODEL_NAME)
    print(f"Model '{SAVE_MODEL_NAME}' loaded successfully.")

    # Create some dummy new data for prediction demonstration
    # In reality, you'd load actual new data: new_data = pd.read_csv('new_unseen_data.csv')
    # Ensure the dummy data has the same columns (except target) as the training data
    if not data_subset.empty:
         # Get columns excluding the target
        feature_columns = data_subset.drop(columns=[TARGET_COLUMN]).columns
        # Create one row of dummy data using mean for numeric and mode for categoricals/objects
        dummy_data_dict = {}
        original_data_for_types = data_subset.drop(columns=[TARGET_COLUMN]) # Use original before setup changes types potentially
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(original_data_for_types[col].dtype):
                dummy_data_dict[col] = [original_data_for_types[col].mean()]
            else:
                dummy_data_dict[col] = [original_data_for_types[col].mode()[0]] # Take first mode if multiple exist

        new_data = pd.DataFrame(dummy_data_dict)

        print("\nExample new data row for prediction:")
        print(new_data)

        predictions = pycaret_module.predict_model(loaded_pipeline, data=new_data)

        print("\nPredictions on new data:")
        # The output DataFrame includes original features + 'prediction_label' (clf) or 'prediction_label' (reg)
        print(predictions)
    else:
        print("Skipping prediction example as initial data was empty.")

except FileNotFoundError:
    print(f"Could not load saved model '{SAVE_MODEL_NAME}.pkl'. Make sure it was saved correctly.")
except Exception as e:
    print(f"An error occurred during the prediction example: {e}")