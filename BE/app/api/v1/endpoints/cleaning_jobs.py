from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.api.v1.dependencies import get_db
from app.schemas.cleaning_jobs import (
    CleaningConfig, CleaningPreview,
    CleaningStatus, CleaningResults, CleaningJobOut, CleaningDataPreview
)
from app.services.cleaning_service import (
    schedule_cleaning, run_cleaning_job,
    preview_cleaning, get_status, get_results,
    update_cleaning, delete_cleaning
)

from app.crud.cleaning_jobs import crud_cleaning_job
from app.utils.file_storage import load_csv_as_dataframe, get_file_path
from app import schemas
import pandas as pd

# --- API Router Setup ---
# Create an API router instance specifically for data cleaning endpoints.
# All routes defined here will be prefixed with '/v1' and tagged as 'cleaning' in OpenAPI docs.
router = APIRouter(prefix="/v1", tags=["cleaning"])


# --- Endpoint Definitions ---

@router.get("/datasets/{dataset_id}/cleaning/preview", response_model=CleaningPreview)
def preview_issues(dataset_id: int, db: Session = Depends(get_db)):
    """
    **Preview Potential Data Quality Issues**

    Analyzes a dataset and provides a preview of potential data quality issues
    (e.g., missing values, outliers, data type inconsistencies) without performing
    any actual cleaning.

    Args:
        `dataset_id` (int): The unique identifier of the dataset to preview.
        `db` (Session): **Dependency:** Database session managed by FastAPI.

    Returns:
        `CleaningPreview`: An object containing summarized information about
                           detected data quality issues.
    """
    # Delegate the preview generation task to the cleaning service
    return preview_cleaning(dataset_id, db)


@router.post("/datasets/{dataset_id}/cleaning", response_model=CleaningJobOut)
def post_cleaning(
    dataset_id: int,
    config: CleaningConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    **Schedule a Data Cleaning Job**

    Creates and schedules a new data cleaning job based on the provided configuration.
    The cleaning process itself is executed asynchronously in the background.

    Args:
        `dataset_id` (int): The unique identifier of the dataset to be cleaned.
        `config` (CleaningConfig): Pydantic model containing the cleaning parameters
                                  (e.g., strategies for handling missing values, duplicates).
        `background_tasks` (BackgroundTasks): **Dependency:** FastAPI utility to run tasks
                                            (like the cleaning job) after the response is sent.
        `db` (Session): **Dependency:** Database session.

    Returns:
        `CleaningJobOut`: Details of the newly created and scheduled cleaning job,
                          including its `job_id`.
    """
    # 1. Schedule the job using the service layer (creates DB record, sets status)
    job_id = schedule_cleaning(dataset_id, config.dict(), db)

    # 2. Add the actual, potentially long-running, cleaning task to the background queue
    background_tasks.add_task(run_cleaning_job, job_id)

    # 3. Retrieve the job details from the database using the generated job_id
    job = crud_cleaning_job.get(db, job_id)

    # 4. Return the job details to the client immediately
    return job


@router.get("/cleaning/{job_id}/status", response_model=CleaningStatus)
def cleaning_status(job_id: int, db: Session = Depends(get_db)):
    """
    **Get Cleaning Job Status**

    Retrieves the current processing status of a specific data cleaning job.

    Args:
        `job_id` (int): The unique identifier of the cleaning job.
        `db` (Session): **Dependency:** Database session.

    Raises:
        `HTTPException`: Status code `404` (Not Found) if a job with the
                         specified `job_id` does not exist.

    Returns:
        `CleaningStatus`: An object containing the current status string
                          (e.g., "PENDING", "RUNNING", "COMPLETED", "FAILED").
    """
    # Fetch the status string from the service layer
    status = get_status(job_id, db)

    # If the service returns None, the job doesn't exist
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Return the status in the expected response model format
    return {"status": status}


@router.get("/cleaning/{job_id}/results", response_model=CleaningResults)
def cleaning_results(job_id: int, db: Session = Depends(get_db)):
    """
    **Get Cleaning Job Results Summary**

    Retrieves a summary of the results for a completed data cleaning job.
    This typically includes statistics about the cleaning process (e.g., rows removed,
    values imputed).

    Args:
        `job_id` (int): The unique identifier of the cleaning job.
        `db` (Session): **Dependency:** Database session.

    Raises:
        `HTTPException`: Status code `404` (Not Found) if results for the job
                         are not available (either the job doesn't exist,
                         is not yet completed, or failed).

    Returns:
        `CleaningResults`: An object containing the summary results of the
                           cleaning process.
    """
    # Fetch the results dictionary from the service layer
    results = get_results(job_id, db)

    # If the service returns None, results aren't ready or the job doesn't exist
    if results is None:
        raise HTTPException(status_code=404, detail="Results not found")

    # Combine the job_id with the results dictionary for the response
    return {"job_id": job_id, **results}


@router.get("/datasets/{dataset_id}/cleaned_data", response_model=CleaningDataPreview)
def get_cleaned_data(dataset_id: int, db: Session = Depends(get_db)):
    """
    **Get Preview of Cleaned Data**

    Retrieves a preview (first N rows) of the cleaned dataset generated by a
    successful cleaning job associated with the original `dataset_id`. Assumes
    the cleaned data is stored in a file following a convention (e.g., `dataset_{id}_cleaned.csv`).

    Args:
        `dataset_id` (int): The unique identifier of the *original* dataset for which
                            cleaned data is requested.
        `db` (Session): **Dependency:** Database session (potentially used by underlying
                        functions, though not directly here).

    Raises:
        `HTTPException`: Status code `500` (Internal Server Error) if the cleaned data
                         file can be loaded but fails during conversion to the response format.
        `HTTPException`: (Implicitly via `load_csv_as_dataframe`) Potential error if the
                         expected cleaned data file cannot be found or loaded.

    Returns:
        `CleaningDataPreview`: An object containing:
            - `preview_cleaned`: A structured preview of the first rows of cleaned data.
            - `preview_row`: The number of rows included in the preview.
            - `total_row`: The total number of rows in the cleaned dataset.
    """
    # Construct the expected file path for the cleaned data CSV
    file_path = get_file_path(f'dataset_{dataset_id}_cleaned.csv')

    # Load the entire cleaned dataset from the CSV file
    result_df = load_csv_as_dataframe(file_path) # Might raise error if file not found

    # Determine number of rows for the preview (capped at 100)
    preview_row = min(100, len(result_df))
    # Select the top rows for the preview
    preview_df = result_df.head(preview_row).copy()
    # Get the total row count from the full dataframe
    total_row = len(result_df)

    try:
        # --- Prepare DataFrame for JSON serialization ---
        # Ensure a clean default index
        df_for_schema = preview_df.reset_index(drop=True)
        # Convert DataFrame to list of lists, replacing Pandas/NumPy NaNs/NaTs with Python's None
        data_list = df_for_schema.astype(object).where(pd.notnull(df_for_schema), None).values.tolist()

        # --- Create the Pydantic schema object for the preview data ---
        preview_schema = schemas.DataFrameStructure(
            columns=df_for_schema.columns.tolist(),
            data=data_list
        )
    except Exception as convert_e:
        # Log error internally for diagnostics (optional)
        print(f"Error converting preview DataFrame to schema: {convert_e}")
        # Raise server error if conversion fails
        raise HTTPException(status_code=500, detail="Failed to format cleaned data preview.")

    # --- Construct the final response object ---
    response_data = CleaningDataPreview(
        preview_cleaned=preview_schema,
        preview_row=preview_row,
        total_row=total_row
    )

    return response_data


@router.put("/cleaning/{job_id}")
def put_cleaning(job_id: int, config: CleaningConfig, db: Session = Depends(get_db)):
    """
    **Update Cleaning Job Configuration**

    Updates the configuration of an *existing* cleaning job.
    *Note: The practical effect depends on the job's state. This typically only
    affects jobs that haven't started running yet.*

    Args:
        `job_id` (int): The unique identifier of the cleaning job to update.
        `config` (CleaningConfig): Pydantic model containing the *new* configuration settings.
        `db` (Session): **Dependency:** Database session.

    Raises:
        `HTTPException`: Status code `404` (Not Found) if a job with the
                         specified `job_id` does not exist.

    Returns:
        `dict`: A simple confirmation message: `{"detail": "Config updated"}`.
    """
    # Attempt the update via the service layer
    success = update_cleaning(job_id, config.dict(), db)

    # If the service returns False, the job wasn't found
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")

    # Return a success confirmation
    return {"detail": "Config updated"}


@router.delete("/cleaning/{job_id}")
def delete_clean(job_id: int, db: Session = Depends(get_db)):
    """
    **Delete Cleaning Job**

    Removes a specific cleaning job record from the system (database).
    *Note: This typically doesn't delete the resulting cleaned data file, only the job metadata.*

    Args:
        `job_id` (int): The unique identifier of the cleaning job to delete.
        `db` (Session): **Dependency:** Database session.

    Raises:
        `HTTPException`: Status code `404` (Not Found) if a job with the
                         specified `job_id` does not exist.

    Returns:
        `dict`: A simple confirmation message: `{"detail": "Job deleted"}`.
    """
    # Attempt the deletion via the service layer
    success = delete_cleaning(job_id, db)

    # If the service returns False, the job wasn't found
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")

    # Return a success confirmation
    return {"detail": "Job deleted"}