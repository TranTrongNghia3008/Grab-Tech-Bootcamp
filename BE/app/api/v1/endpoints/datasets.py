from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
import io
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
import pandas as pd
from app.crud.datasets import DatasetCreateSchema, DatasetUpdateSchema
from app.schemas.dataset import DatasetFromConnection, DatasetId, DatasetPreview

# Import CRUD functions for datasets and connections
from app.crud.datasets import crud_dataset
from app.crud.connections import get_connection_by_id

# Import common dependencies and utilities
from app.api.v1.dependencies import get_db
from app.utils.file_storage import save_dataframe_as_csv, get_file_path, load_csv_as_dataframe
from app import schemas

# --- API Router Setup ---
router = APIRouter(
    prefix='/v1',
    tags=['datasets']
)

# --- Endpoint Definitions ---

@router.post('/datasets/', response_model=DatasetId)
async def upload_datasets(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    **Upload Dataset from CSV File**

    Initiates the process of creating a new dataset resource from an uploaded CSV file.

    This endpoint performs the following actions sequentially:
    1.  Validates that the uploaded filename ends with `.csv`.
    2.  Asynchronously reads the raw byte content of the uploaded file.
    3.  Decodes the file content (assuming UTF-8).
    4.  Parses the decoded content into a pandas DataFrame using `pd.read_csv`.
    5.  Creates an initial `Dataset` record in the database (without a file path yet) via `crud_dataset.create`.
    6.  Constructs a filename for storage using the newly generated dataset ID (e.g., `dataset_{id}.csv`).
    7.  Saves the pandas DataFrame to the configured file storage as a CSV file using `save_dataframe_as_csv`.
        *   If file saving fails, the previously created database record (Step 5) is deleted (rollback).
    8.  Updates the `Dataset` record in the database with the actual path to the saved file via `crud_dataset.update`.
    9.  Returns the unique ID (`id`) of the successfully created dataset record.

    Args:
        `file` (UploadFile): The CSV file being uploaded via form-data.
        `db` (Session): **Dependency:** Injected database session.

    Raises:
        `HTTPException`:
            - `400`: If the filename does not end with `.csv`.
            - `400`: If the CSV file content cannot be decoded or parsed by pandas.
            - `500`: If saving the DataFrame to the file storage fails after the initial DB record creation.

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
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")

    # 5. Create Initial Database Record
    dataset_in_create = DatasetCreateSchema(connection_id=None)
    ds = crud_dataset.create(db=db, obj_in=dataset_in_create)

    # 6. Construct filename
    filename = f"dataset_{ds.id}.csv"

    # 7. Save DataFrame to File Storage (with rollback)
    try:
        path = save_dataframe_as_csv(df, filename)
    except Exception as e:
        # Cleanup the created DB record if file saving fails
        crud_dataset.remove(db=db, id=ds.id)
        raise HTTPException(status_code=500, detail=f"Failed to save dataset file: {e}")

    # 8. Update Database Record with File Path
    dataset_in_update = DatasetUpdateSchema(file_path=path)
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
def get_dataset_preview(dataset_id: int):
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
    total_row = len(result_df)

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
        total_row=total_row
    )

    # 9. Return Response
    return response_data