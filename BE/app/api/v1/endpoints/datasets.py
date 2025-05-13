from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Query, status, Body
import io
from typing import List, Optional, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine
import pandas as pd
from app.crud.datasets import DatasetCreateSchema, DatasetUpdateSchema
from app.schemas.dataset import (
    DatasetFromConnection, 
    DatasetId, 
    DatasetPreview, 
    DatasetFlatListResponse, 
    DatasetAnalysisReport, 
    DatasetFlatInfo, 
    DataPreviewSchema, 
    DetailResponse,
    DatasetUpdateProjectName
)

# Import CRUD functions for datasets and connections
from app.crud.datasets import crud_dataset
from app.crud.connections import get_connection_by_id

# Import common dependencies and utilities
from app.api.v1.dependencies import get_db
from app.utils.file_storage import save_dataframe_as_csv, get_file_path, load_csv_as_dataframe
from app import schemas
from ydata_profiling import ProfileReport
from fastapi.responses import FileResponse, HTMLResponse

# --- API Router Setup ---
router = APIRouter(
    prefix='/v1',
    tags=['datasets']
)

# --- Endpoint Definitions ---

@router.post('/datasets/', response_model=DatasetId)
async def upload_datasets(project_name: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    **Upload Dataset from CSV File with a Project Name**

    Initiates the process of creating a new dataset resource from an uploaded CSV file
    and associates it with a project name.

    This endpoint performs the following actions sequentially:
    1.  Validates that the uploaded filename ends with `.csv`.
    2.  Asynchronously reads the raw byte content of the uploaded file.
    3.  Decodes the file content (assuming UTF-8).
    4.  Parses the decoded content into a pandas DataFrame using `pd.read_csv`.
    5.  Creates an initial `Dataset` record in the database with the provided `project_name`
        (without a file path yet) via `crud_dataset.create`.
        *   If `project_name` is not unique (and the DB enforces it), a 409 Conflict is raised.
    6.  Constructs a filename for storage using the newly generated dataset ID (e.g., `dataset_{id}.csv`).
    7.  Saves the pandas DataFrame to the configured file storage as a CSV file using `save_dataframe_as_csv`.
        *   If file saving fails, the previously created database record (Step 5) is deleted (rollback).
    8.  Updates the `Dataset` record in the database with the actual path to the saved file via `crud_dataset.update`.
    9.  Returns the unique ID (`id`) of the successfully created dataset record.

    Args:
        `project_name` (str): The name of the project for this dataset (sent as form-data).
        `file` (UploadFile): The CSV file being uploaded via form-data.
        `db` (Session): **Dependency:** Injected database session.

    Raises:
        `HTTPException`:
            - `400`: If the filename does not end with `.csv`.
            - `400`: If the CSV file content cannot be decoded or parsed by pandas.
            - `409`: If the `project_name` already exists (and is constrained to be unique).
            - `500`: If saving the DataFrame to the file storage fails after the initial DB record creation.
            - `500`: For other database errors during dataset creation.

    Returns:
        `DatasetId`: An object containing the assigned `id` for the new dataset.
    """
    # 1. Validate File Type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='Only CSV files are supported.')

    # 2, 3, 4. Read and Parse CSV
    try:
        content = await file.read()
        decoded = content.decode('utf-8')
        df = pd.read_csv(io.StringIO(decoded))
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Error decoding file: Ensure it's UTF-8 encoded.")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV file: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")

    # 5. Create Initial Database Record
    dataset_in_create = DatasetCreateSchema(project_name=project_name,connection_id=None)
    ds = None # Initialize ds to None
    try:
        ds = crud_dataset.create(db=db, obj_in=dataset_in_create)
    except IntegrityError as e:
        db.rollback() # Ensure the session is rolled back if your CRUD doesn't handle it
        error_detail = str(e.orig).lower()
        if "unique constraint" in error_detail and "project_name" in error_detail:
            raise HTTPException(status_code=409, detail=f"Project name '{project_name}' already exists.")
        else:
            # For other IntegrityErrors or if the check is too broad
            raise HTTPException(status_code=500, detail=f"Database integrity error: {e}")
    except Exception as e: # Catch other potential errors during DB creation
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating dataset record in DB: {e}")

    if not ds: # Should not happen if no exception was raised, but good for safety
        raise HTTPException(status_code=500, detail="Failed to create dataset record, unknown error.")

    # 6. Construct filename
    filename = f"dataset_{ds.id}.csv"
    filename_cleaned = f"dataset_{ds.id}_cleaned.csv"

    # 7. Save DataFrame to File Storage (with rollback)
    saved_file_path = None
    try:
        saved_file_path = save_dataframe_as_csv(df, filename)
        save_dataframe_as_csv(df, filename_cleaned)
    except Exception as e:
        # Cleanup the created DB record if file saving fails
        crud_dataset.remove(db=db, id=ds.id)
        db.commit() # If remove doesn't commit
        raise HTTPException(status_code=500, detail=f"Failed to save dataset file: {e}")

    # 8. Update Database Record with File Path
    dataset_in_update = DatasetUpdateSchema(file_path=saved_file_path, project_name=ds.project_name) # Pass existing project_name
    ds_updated = crud_dataset.update(db=db, db_obj=ds, obj_in=dataset_in_update)

    # 9. Return the Dataset ID
    return DatasetId(id=ds_updated.id)

@router.post('/datasets/from-connection/', response_model=DatasetId)
async def ingest_from_connection(payload: DatasetFromConnection, db: Session = Depends(get_db)):
    """
    **Ingest Dataset from Database Connection via SQL Query**

    Initiates the process of creating a new dataset resource by executing a SQL query
    against a pre-configured database connection.

    This endpoint performs the following actions sequentially:
    1.  Validates the request body (`payload`) using Pydantic (`DatasetFromConnection`).
    2.  Retrieves the database connection details (`host`, `port`, `user`, etc.) using the `payload.connection_id` via `get_connection_by_id`.
    3.  Validates that the connection details were found.
    4.  Constructs a SQLAlchemy database connection URL string from the retrieved details.
    5.  Creates a temporary SQLAlchemy engine using `create_engine`.
    6.  Connects to the target database using the engine.
    7.  Executes the SQL query provided in `payload.query` using `pd.read_sql` to fetch data into a pandas DataFrame.
    8.  Disposes of the SQLAlchemy engine to release resources (`engine.dispose()`).
    9.  Creates an initial `Dataset` record in the database, linking it to the `payload.connection_id`, via `crud_dataset.create`.
    10. Constructs a filename for storage using the newly generated dataset ID (e.g., `dataset_{id}.csv`).
    11. Saves the fetched pandas DataFrame to the configured file storage as a CSV file using `save_dataframe_as_csv`.
        *   If file saving fails, the previously created database record (Step 9) is deleted (rollback).
    12. Updates the `Dataset` record in the database with the actual path to the saved file via `crud_dataset.update`.
    13. Returns the unique ID (`id`) of the successfully created dataset record.

    Args:
        `payload` (DatasetFromConnection): Request body containing:
            - `connection_id` (int): ID of the pre-configured database connection.
            - `query` (str): The SQL query to execute.
        `db` (Session): **Dependency:** Injected database session.

    Raises:
        `HTTPException`:
            - `404`: If the `connection_id` is not found.
            - `500`: If constructing the database URL fails (e.g., missing connection details).
            - `400`: If connecting to the database or executing the SQL query fails.
            - `500`: If saving the fetched DataFrame to file storage fails after initial DB record creation.

    Returns:
        `DatasetId`: An object containing the assigned `id` for the new dataset.
    """
    # 1. Validation (Implicit by Pydantic)

    # 2. Retrieve Connection Details
    conn = get_connection_by_id(db, payload.connection_id)
    # 3. Validate connection exists
    if not conn:
        raise HTTPException(status_code=404, detail='Connection not found')

    # 4. Construct Database URL
    try:
        db_url = (
            f'{conn.type}://{conn.username}:{conn.password}@'
            f'{conn.host}:{conn.port}/{conn.database}'
        )
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error building database URL: {e}")

    # 5, 6, 7, 8. Execute Query and Fetch Data (with resource disposal)
    engine = None # Ensure engine is defined for finally block
    try:
        engine = create_engine(db_url)
        with engine.connect() as connection:
             df = pd.read_sql(payload.query, con=connection)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Failed to execute query or read data: {e}')
    finally:
        if engine:
            engine.dispose() # Release connection pool resources

    # 9. Create initial DB record
    dataset_in_create = DatasetCreateSchema(connection_id=payload.connection_id)
    ds = crud_dataset.create(db=db, obj_in=dataset_in_create)

    # 10. Construct filename
    filename = f"dataset_{ds.id}.csv"
    # 11. Save the dataframe (with rollback)
    try:
        path = save_dataframe_as_csv(df, filename)
    except Exception as e:
        # Cleanup DB record if saving fails
        crud_dataset.remove(db=db, id=ds.id)
        raise HTTPException(status_code=500, detail=f"Failed to save dataset file: {e}")

    # 12. Update the DB record with the path
    dataset_in_update = DatasetUpdateSchema(file_path=path)
    ds_updated = crud_dataset.update(db=db, db_obj=ds, obj_in=dataset_in_update)

    # 13. Return the Dataset ID
    return DatasetId(id=ds_updated.id)

@router.get('/datasets/{dataset_id}/preview', response_model=DatasetPreview)
def get_dataset_preview(dataset_id: int, db: Session = Depends(get_db)):
    """
    **Get Preview of (Cleaned) Dataset Data**

    Retrieves a preview (typically the first 100 rows) of the data associated with a dataset ID.
    **Note:** This implementation specifically loads the *cleaned* data file (`_cleaned.csv`).

    This endpoint performs the following actions sequentially:
    1.  Constructs the expected file path for the *cleaned* dataset file (e.g., `storage/dataset_{id}_cleaned.csv`).
    2.  Loads the entire dataset from the CSV file at the constructed path into a pandas DataFrame using `load_csv_as_dataframe`.
    3.  Determines the number of rows for the preview (minimum of 100 or total rows if fewer).
    4.  Extracts the first `preview_row` rows from the DataFrame using `.head()`.
    5.  Gets the total number of rows from the originally loaded DataFrame (`len(result_df)`).
    6.  Formats the preview DataFrame for JSON serialization:
        *   Resets the index.
        *   Converts the DataFrame to a list of lists using `.values.tolist()`.
        *   Replaces any `NaN`/`NaT` values with `None` using `.astype(object).where(pd.notnull(...), None)`.
    7.  Constructs a `schemas.DataFrameStructure` object containing the preview columns and data list.
    8.  Constructs the final `DatasetPreview` response object, including the preview structure, preview row count, and total row count.
    9.  Returns the `DatasetPreview` object.

    Args:
        `dataset_id` (int): The unique identifier of the dataset.

    Raises:
        `HTTPException`:
            - Potentially (from `load_csv_as_dataframe`): If the expected cleaned data file (`dataset_{id}_cleaned.csv`) is not found or cannot be loaded.
            - `500`: If an error occurs during the formatting of the preview data for the JSON response.

    Returns:
        `DatasetPreview`: An object containing the data preview structure, preview row count, and total row count.
    """
    # 1. Determine File Path (for cleaned data)
    file_path = get_file_path(f'dataset_{dataset_id}_cleaned.csv')

    # 2. Load Data
    result_df = load_csv_as_dataframe(file_path)

    # 3. Calculate Preview Rows
    preview_row = min(100, len(result_df))
    # 4. Extract Preview DataFrame
    preview_df = result_df.head(preview_row).copy()
    # 5. Get Total Rows
    total_row, total_col = result_df.shape
    
    project_name = crud_dataset.get(db, id=dataset_id).project_name

    # 6, 7. Format Preview for Response
    try:
        df_for_schema = preview_df.reset_index(drop=True)
        data_list = df_for_schema.astype(object).where(pd.notnull(df_for_schema), None).values.tolist()
        preview_schema = schemas.DataFrameStructure(
            columns=df_for_schema.columns.tolist(),
            data=data_list
        )
    except Exception as convert_e:
        print(f"Error converting preview DataFrame to schema: {convert_e}")
        raise HTTPException(status_code=500, detail="Failed to format dataset preview data.") # Corrected detail message

    # 8. Construct Response Object
    response_data = DatasetPreview(
        preview_data=preview_schema,
        preview_row=preview_row,
        total_row=total_row,
        total_col=total_col,
        project_name=project_name
    )

    # 9. Return Response
    return response_data

@router.get(
    "/datasets/all-by-creation/",
    response_model=DatasetFlatListResponse,
    summary="List ALL Datasets Ordered by Creation Date (Newest First)"
)
async def list_all_datasets_ordered_by_creation_date(
    db: Session = Depends(get_db)
    # Performance note: For endpoints listing ALL items, especially with I/O per item,
    # consider pagination (e.g., adding skip: int = 0, limit: int = 100 parameters).
):
    """
    **Retrieve a flat list of ALL datasets, ordered by their creation date (newest first).**

    Each dataset entry includes:
    - `id`: The dataset's unique identifier.
    - `project_name`: The name of the project the dataset belongs to (if any).
    - `created_at`: The timestamp when the dataset record was created.
    - `is_model`: Check if the dataset was trained or not.
    - `is_clean`: Check if the dataset was cleaned or not.
    - `data_preview`: A preview of the dataset (first 4 rows and 4 columns).
                      Format: {"columns": ["col1", ...], "data": [[val1, ...], ...]}.
                      Preview is from the cleaned version of the dataset if available, otherwise original.
                      Will be null if data cannot be loaded, is empty, or file path is missing.
    """
    db_dataset_models = crud_dataset.get_all_datasets_ordered_by_creation(db=db, descending=True)

    response_datasets: List[DatasetFlatInfo] = []
    for dataset_model in db_dataset_models:
        data_preview_payload: Optional[DataPreviewSchema] = None
        df: Optional[pd.DataFrame] = None

        # --- Data Loading for Preview ---
        # This logic attempts to mirror crud_dataset.get_dataset_analysis:
        # 1. Try to load cleaned data (convention: dataset_{id}_cleaned.csv).
        # 2. If not found, fall back to the original dataset file_path.
        # Assumes dataset_model has 'id' and 'file_path' attributes.

        model_id = getattr(dataset_model, 'id', None)
        original_file_path = getattr(dataset_model, 'file_path', None)

        if model_id is not None and original_file_path and isinstance(original_file_path, str):
            try:
                cleaned_filename_candidate = f"dataset_{model_id}_cleaned.csv"
                # get_file_path needs to be robust and find files based on your storage setup.
                potential_cleaned_path = get_file_path(cleaned_filename_candidate)
                df = load_csv_as_dataframe(potential_cleaned_path)
            except FileNotFoundError:
                try:
                    df = load_csv_as_dataframe(original_file_path)
                except FileNotFoundError:
                    df = None # File not found, df remains None
                except Exception as e_orig: # Other errors loading original file
                    df = None
            except Exception as e_cleaned: # Other errors loading cleaned file (permissions, corrupt, etc.)
                try:
                    df = load_csv_as_dataframe(original_file_path)
                except Exception as e_orig_fallback:
                    df = None
        elif original_file_path and isinstance(original_file_path, str): # Fallback if no ID, but path exists
             try:
                df = load_csv_as_dataframe(original_file_path)
             except Exception as e_load_basic:
                df = None
        # If df is still None, it means no file could be loaded or paths were invalid.

        # --- Preview Generation ---
        if df is not None and not df.empty:
            try:
                num_rows_to_take = min(4, df.shape[0])
                num_cols_to_take = min(4, df.shape[1])
                
                preview_df_slice = df.iloc[:num_rows_to_take, :num_cols_to_take]
                
                # Ensure column names are strings
                column_names = [str(col) for col in preview_df_slice.columns.tolist()]

                serializable_data_rows: List[List[Any]] = []
                # iterrows is not the most performant for large DFs, but fine for a small 4x4 slice.
                for _, row_series in preview_df_slice.iterrows():
                    processed_row: List[Any] = []
                    for item in row_series:
                        if pd.isna(item): # Handle NaN/NaT
                            processed_row.append(None)
                        elif isinstance(item, (datetime, pd.Timestamp)): # Convert datetime objects to ISO strings
                            processed_row.append(item.isoformat())
                        else:
                            processed_row.append(item) # Pass through other types
                    serializable_data_rows.append(processed_row)

                data_preview_payload = DataPreviewSchema(
                    columns=column_names,
                    data=serializable_data_rows
                )
            except Exception as e_preview_gen:
                data_preview_payload = None 

        dataset_flat_info = DatasetFlatInfo(
            id=dataset_model.id,
            project_name=getattr(dataset_model, 'project_name', None),
            created_at=dataset_model.created_at,
            is_model=getattr(dataset_model, 'is_model', False),
            is_clean=getattr(dataset_model, 'is_clean', False),
            data_preview=data_preview_payload
        )
        response_datasets.append(dataset_flat_info)
    
    return DatasetFlatListResponse(datasets=response_datasets)

@router.get(
    "/datasets/{dataset_id}/analysis-report/",
    response_model=DatasetAnalysisReport,
    summary="Get Detailed Analysis Report for a Dataset"
)
async def get_dataset_analysis_report(dataset_id: int, db: Session = Depends(get_db)):
    """
    **Retrieve a detailed analysis report for a specific dataset.**

    The report includes:
    - Overall dataset statistics: total records, total features, overall missing percentage.
    - A data quality score (currently based on completeness).
    - A list of all features (columns) with:
        - `feature_name`: Name of the column.
        - `dtype`: Inferred data type (e.g., integer, float, datetime, categorical/string).
        - `missing_percentage`: Percentage of missing values in that column.
        - `unique_percentage`: Percentage of unique values in that column relative to total records.

    **Note:** The analysis attempts to use the 'cleaned' version of the dataset if available
    (e.g., `dataset_{id}_cleaned.csv`), otherwise it uses the primary dataset file.
    """
    try:
        report = crud_dataset.get_dataset_analysis(db=db, dataset_id=dataset_id)
        if report is None: # If crud_dataset.get itself returned None for not found dataset
            raise HTTPException(status_code=404, detail=f"Dataset with ID {dataset_id} not found.")
        return report
    except FileNotFoundError as e:
        # This might be raised from load_csv_as_dataframe if file doesn't exist
        raise HTTPException(status_code=404, detail=f"Dataset file for ID {dataset_id} not found: {e}")
    except ValueError as e: # e.g., if dataset has no file_path from CRUD function
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e: # For other loading/processing errors from CRUD function
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Catch-all for unexpected errors
        # Log this error for debugging
        print(f"Unexpected error during dataset analysis for ID {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while analyzing the dataset.")
    
@router.delete(
    "/datasets/{dataset_id}",
    response_model=DetailResponse, # Response is a simple message
    status_code=status.HTTP_200_OK, # Or 204 if you prefer no content
    summary="Delete a dataset",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Dataset not found"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error during deletion"},
        # Add others if applicable (e.g., 403 Forbidden)
    }
)
def delete_dataset(
    *,
    db: Session = Depends(get_db),
    dataset_id: int
) -> DetailResponse:
    """
    Deletes a dataset specified by its ID.

    This will also delete associated records if cascade options are set
    (e.g., CleaningJob, AutoMLSession, ChatSessionState).
    """

    # Use the remove method from CRUDBase via crud_dataset instance
    deleted_dataset = crud_dataset.remove(db=db, id=dataset_id)

    if not deleted_dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with id {dataset_id} not found."
        )

    return DetailResponse(detail=f"Dataset with id {dataset_id} deleted successfully.")

@router.patch(
    "/datasets/{dataset_id}/project_name", # More specific path for updating just the name
    response_model=DatasetFlatInfo, # Return updated dataset info
    status_code=status.HTTP_200_OK,
    summary="Update dataset project name",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Dataset not found"},
        status.HTTP_409_CONFLICT: {"description": "Project name already exists"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error (e.g., empty string)"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal server error during update"},
    }
)
def update_dataset_project_name(
    *,
    db: Session = Depends(get_db),
    dataset_id: int,
    update_data: DatasetUpdateProjectName = Body(...) # Get data from request body
) -> DatasetFlatInfo:
    """
    Updates the 'project_name' for a specific dataset.
    The new name must be unique across all datasets if it is not null.
    """

    # Use the specific CRUD method
    updated_dataset, error_msg = crud_dataset.update_project_name(
        db=db,
        dataset_id=dataset_id,
        new_project_name=update_data.project_name
    )

    if error_msg:
        # Check if the dataset itself was found but validation failed
        if updated_dataset is not None: # Dataset exists, but name conflict or DB error
             if "already exists" in error_msg:
                 status_code = status.HTTP_409_CONFLICT
             else: # Other database error during commit
                 status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
             raise HTTPException(status_code=status_code, detail=error_msg)
        else:
              raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Inconsistent server state during update.")


    if updated_dataset is None:
        # This means the initial get in update_project_name returned None -> dataset not found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with id {dataset_id} not found."
        )

    return updated_dataset

@router.get(
    "/datasets/{dataset_id}/profile/download",
    summary="Generate and download ydata-profiling report",
    responses={
        status.HTTP_200_OK: {
            "content": {"text/html": {}},
            "description": "Returns the ydata-profiling report as an HTML file attachment.",
        },
        status.HTTP_404_NOT_FOUND: {"description": "Dataset not found or source file missing"},
        status.HTTP_409_CONFLICT: {"description": "Dataset found but has no associated file path"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Error loading data or generating profile"},
    }
)
def download_dataset_profile(
    *,
    db: Session = Depends(get_db),
    dataset_id: int
) -> HTMLResponse: # Return HTMLResponse directly
    """
    Generates a ydata-profiling report for the specified dataset
    and returns it as an HTML file for download.

    *Note: This can be resource-intensive for large datasets.*
    """

    # 1. Get Dataset information
    db_dataset = crud_dataset.get(db=db, id=dataset_id)
    if not db_dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with id {dataset_id} not found."
        )

    if not db_dataset.file_path:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, # Conflict state: dataset exists but unusable for this
            detail=f"Dataset {dataset_id} does not have a source file path configured."
        )

    # 2. Load DataFrame
    try:
        df = load_csv_as_dataframe(db_dataset.file_path)
        if df.empty:
             # You might want to return a specific message or an empty report
             raise HTTPException(
                 status_code=status.HTTP_400_BAD_REQUEST, # Or maybe 200 with an empty report message?
                 detail=f"The dataset file for id {dataset_id} is empty."
             )

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, # Treat missing file as Not Found resource dependency
            detail=f"Source data file not found for dataset id {dataset_id}."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load data for dataset id {dataset_id}."
        )

    # 3. Generate Profile Report (consider options for large data)
    try:
        profile = ProfileReport(
            df,
            title=f"Dataset Profile: {db_dataset.project_name or f'ID {dataset_id}'}",
        )

        # Generate the HTML content directly
        html_report_content = profile.to_html()

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate profile report for dataset id {dataset_id}."
        )

    # 4. Prepare and Return Response
    # Define headers for download
    file_name = f"dataset_{dataset_id}_profile.html"
    headers = {
        "Content-Disposition": f"attachment; filename=\"{file_name}\""
    }

    # Return the generated HTML content using HTMLResponse
    return HTMLResponse(
        content=html_report_content,
        status_code=status.HTTP_200_OK,
        headers=headers,
        media_type="text/html" # Explicitly set media type
    )