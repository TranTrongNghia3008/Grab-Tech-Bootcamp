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
# Create an API router instance specifically for dataset management endpoints.
# All routes defined here will be prefixed with '/v1' and tagged as 'datasets' in OpenAPI docs.
router = APIRouter(
    prefix='/v1',
    tags=['datasets']
)

# --- Endpoint Definitions ---

@router.post('/datasets/', response_model=DatasetId)
async def upload_datasets(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    **Upload Dataset from CSV File**

    Allows users to upload a dataset directly as a CSV file. The file is read,
    saved to the configured storage, and a corresponding record is created in the database.

    Args:
        `file` (UploadFile): The CSV file being uploaded. Must have a `.csv` extension.
        `db` (Session): **Dependency:** Database session managed by FastAPI.

    Raises:
        `HTTPException`:
            - `400` (Bad Request): If the uploaded file is not a CSV (`.csv` extension missing).
            - `400` (Bad Request): If the CSV file cannot be read or parsed by pandas.
            - `500` (Internal Server Error): If saving the parsed data to the file storage fails.

    Returns:
        `DatasetId`: An object containing the unique `id` assigned to the newly created dataset record.
    """
    # 1. **Validate File Type**: Ensure the uploaded file has a '.csv' extension.
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='Only CSV files are supported.')

    # 2. **Read and Parse CSV**:
    try:
        # Read the file content asynchronously
        content = await file.read()
        # Decode assuming UTF-8 encoding
        decoded = content.decode('utf-8')
        # Use pandas to read the decoded string content into a DataFrame
        df = pd.read_csv(io.StringIO(decoded))
    except Exception as e:
        # Handle potential errors during reading or parsing (e.g., bad format, encoding issues)
        raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")

    # 3. **Create Initial Database Record**:
    # Create a basic dataset record in the database first. We need the ID for the filename.
    # At this stage, it doesn't have a file path yet.
    dataset_in_create = DatasetCreateSchema(connection_id=None) # No connection ID for file uploads
    ds = crud_dataset.create(db=db, obj_in=dataset_in_create)

    # 4. **Save DataFrame to File Storage**:
    # Construct a filename incorporating the unique dataset ID.
    filename = f"dataset_{ds.id}.csv"
    try:
        # Use the utility function to save the DataFrame as a CSV file.
        path = save_dataframe_as_csv(df, filename)
    except Exception as e:
        # **Critical**: If saving the file fails, we must roll back the database record creation.
        crud_dataset.remove(db=db, id=ds.id)
        raise HTTPException(status_code=500, detail=f"Failed to save dataset file: {e}")

    # 5. **Update Database Record with File Path**:
    # Now that the file is saved successfully, update the dataset record with the actual file path.
    dataset_in_update = DatasetUpdateSchema(file_path=path)
    ds_updated = crud_dataset.update(db=db, db_obj=ds, obj_in=dataset_in_update)

    # 6. **Return the Dataset ID**:
    # Respond with the ID of the successfully created and saved dataset.
    return DatasetId(id=ds_updated.id)

@router.post('/datasets/from-connection/', response_model=DatasetId)
async def ingest_from_connection(payload: DatasetFromConnection, db: Session = Depends(get_db)):
    """
    **Ingest Dataset from Database Connection**

    Creates a new dataset by executing a SQL query against a pre-configured database connection.
    The results of the query are fetched, saved as a CSV file, and a dataset record is created.

    Args:
        `payload` (DatasetFromConnection): Pydantic model containing:
            - `connection_id` (int): The ID of the pre-configured database connection to use.
            - `query` (str): The SQL query to execute for fetching data.
        `db` (Session): **Dependency:** Database session.

    Raises:
        `HTTPException`:
            - `404` (Not Found): If the specified `connection_id` does not exist.
            - `500` (Internal Server Error): If building the database connection URL fails.
            - `400` (Bad Request): If the SQL query execution fails or data cannot be read.
            - `500` (Internal Server Error): If saving the fetched data to a file fails.

    Returns:
        `DatasetId`: An object containing the unique `id` assigned to the newly created dataset record.
    """
    # 1. **Retrieve Connection Details**:
    # Fetch the connection details (credentials, host, etc.) using the provided ID.
    conn = get_connection_by_id(db, payload.connection_id)
    if not conn:
        raise HTTPException(status_code=404, detail='Connection not found')

    # 2. **Construct Database URL**:
    # Build the SQLAlchemy database connection URL string.
    try:
        db_url = (
            f'{conn.type}://{conn.username}:{conn.password}@'
            f'{conn.host}:{conn.port}/{conn.database}'
        )
    except Exception as e:
         # Catch potential issues if connection details are malformed/missing
         raise HTTPException(status_code=500, detail=f"Error building database URL: {e}")

    # 3. **Execute Query and Fetch Data**:
    engine = None # Ensure engine is defined for the finally block
    try:
        # Create a SQLAlchemy engine using the constructed URL.
        engine = create_engine(db_url)
        # Connect to the database and execute the provided SQL query using pandas.
        with engine.connect() as connection:
             df = pd.read_sql(payload.query, con=connection)
    except Exception as e:
        # Handle errors during connection, query execution, or data reading.
        raise HTTPException(status_code=400, detail=f'Failed to execute query or read data: {e}')
    finally:
        # **Important**: Ensure database connection resources are released.
        if engine:
            engine.dispose()

    # 4. **Create Initial Database Record**:
    # Create the dataset record, linking it to the connection ID used.
    dataset_in_create = DatasetCreateSchema(connection_id=payload.connection_id)
    ds = crud_dataset.create(db=db, obj_in=dataset_in_create)

    # 5. **Save DataFrame to File Storage**:
    # Generate filename and save the fetched data as a CSV.
    filename = f"dataset_{ds.id}.csv"
    try:
        path = save_dataframe_as_csv(df, filename)
    except Exception as e:
        # **Critical**: Rollback DB record if file saving fails.
        crud_dataset.remove(db=db, id=ds.id)
        raise HTTPException(status_code=500, detail=f"Failed to save dataset file: {e}")

    # 6. **Update Database Record with File Path**:
    # Update the record with the path to the saved CSV file.
    dataset_in_update = DatasetUpdateSchema(file_path=path)
    ds_updated = crud_dataset.update(db=db, db_obj=ds, obj_in=dataset_in_update)

    # 7. **Return the Dataset ID**:
    return DatasetId(id=ds_updated.id)


@router.get('/datasets/{dataset_id}/preview', response_model=DatasetPreview)
def get_dataset_preview(dataset_id: int):
    """
    **Get Preview of (Cleaned) Dataset Data**

    Retrieves a preview (first N rows, typically 100) of the data associated with
    a given dataset ID. **Note:** This specific implementation currently attempts
    to load the *cleaned* version of the dataset (`_cleaned.csv`).

    Args:
        `dataset_id` (int): The unique identifier of the dataset whose preview is needed.

    Raises:
        `HTTPException`:
            - (Implicit via `load_csv_as_dataframe`): If the data file (`dataset_{id}_cleaned.csv`)
              cannot be found or loaded.
            - `500` (Internal Server Error): If the loaded data preview cannot be formatted
              correctly for the JSON response.

    Returns:
        `DatasetPreview`: An object containing:
            - `preview_data`: A structured preview of the first rows of data.
            - `preview_row`: The number of rows included in the preview.
            - `total_row`: The total number of rows in the loaded dataset file.
    """
    # 1. **Determine File Path**: Construct the path for the *cleaned* dataset file.
    # !! Important: This assumes a cleaning process has run and produced this file.
    # !! To preview the *original* uploaded data, the filename should be `dataset_{dataset_id}.csv`.
    file_path = get_file_path(f'dataset_{dataset_id}_cleaned.csv')

    # 2. **Load Data**: Load the dataset from the CSV file into a pandas DataFrame.
    result_df = load_csv_as_dataframe(file_path) # May raise error if file not found/invalid

    # 3. **Calculate Preview/Total Rows**: Determine preview size (max 100) and total rows.
    preview_row = min(100, len(result_df))
    preview_df = result_df.head(preview_row).copy() # Take the top rows for the preview
    total_row = len(result_df) # Get total count from the full DataFrame

    # 4. **Format Preview for Response**:
    try:
        # Prepare DataFrame for JSON: Ensure clean index, handle NaN/NaT values.
        df_for_schema = preview_df.reset_index(drop=True)
        # Convert to list of lists, replacing incompatible values (NaN) with None.
        data_list = df_for_schema.astype(object).where(pd.notnull(df_for_schema), None).values.tolist()
        # Create the Pydantic schema object for the structured preview.
        preview_schema = schemas.DataFrameStructure(
            columns=df_for_schema.columns.tolist(),
            data=data_list
        )
    except Exception as convert_e:
        # Log error internally (optional)
        print(f"Error converting preview DataFrame to schema: {convert_e}")
        # Raise server error if formatting fails
        raise HTTPException(status_code=500, detail="Failed to format dataset preview data.")

    # 5. **Construct and Return Response**:
    # Assemble the final response object using the formatted preview and row counts.
    response_data = DatasetPreview(
        preview_data=preview_schema,
        preview_row=preview_row,
        total_row=total_row
    )

    return response_data