from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd # Explicitly import pandas, used by df.describe() and df.corr()

# Import database models and dependencies
from app.api.v1.dependencies import get_db
from app.db.models.datasets import Dataset # Assuming this is the correct path to your SQLAlchemy model
from app.utils.file_storage import load_csv_as_dataframe

# --- API Router Setup ---
# Create an API router instance specifically for Exploratory Data Analysis (EDA) endpoints.
# All routes defined here will be prefixed with '/v1/datasets' and tagged as 'EDA' in OpenAPI docs.
router = APIRouter(
    prefix='/v1/datasets',
    tags=['EDA']
)

# --- Endpoint Definitions ---

@router.get('/{dataset_id}/eda/stats')
def get_summary_statistics(dataset_id: int, db: Session = Depends(get_db)):
    """
    **Get Summary Statistics for a Dataset**

    Calculates descriptive statistics for all columns (both numerical and categorical)
    in the specified dataset. Uses pandas `describe(include='all')`.

    Args:
        `dataset_id` (int): The unique identifier of the dataset.
        `db` (Session): **Dependency:** Database session managed by FastAPI.

    Raises:
        `HTTPException`:
            - `404` (Not Found): If a dataset with the specified `dataset_id` does not exist.
            - (Implicit via `load_csv_as_dataframe`): If the dataset file associated with
              the `dataset_id` cannot be found or loaded.

    Returns:
        `dict`: A dictionary representation of the pandas DataFrame containing summary statistics.
                NaN values in the statistics DataFrame are replaced with empty strings (`''`)
                for better JSON compatibility.
    """
    # 1. **Retrieve Dataset Record**: Fetch the dataset metadata from the database using its ID.
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    # 2. **Validate Dataset Existence**: If no record is found, raise a 404 error.
    if not dataset:
        raise HTTPException(status_code=404, detail='Dataset not found')

    # 3. **Load Dataset Data**: Load the actual data from the CSV file path stored in the dataset record.
    #    `load_csv_as_dataframe` is expected to handle potential file loading errors.
    df = load_csv_as_dataframe(dataset.file_path)

    # 4. **Calculate Summary Statistics**: Use pandas `describe(include='all')` to get statistics
    #    for all columns (numeric types get count, mean, std, min, quantiles, max;
    #    object/categorical types get count, unique, top, freq).
    stats_df = df.describe(include='all')

    # 5. **Format and Return**: Convert the resulting statistics DataFrame to a dictionary.
    #    Replace any remaining NaN values (common in describe output) with empty strings
    #    to avoid potential issues with JSON serialization (though `to_dict` often handles this).
    return stats_df.fillna('').to_dict()


@router.get('/{dataset_id}/eda/corr')
def get_correlation_matrix(dataset_id: int, db: Session = Depends(get_db)):
    """
    **Get Correlation Matrix for a Dataset**

    Calculates the pairwise correlation matrix (e.g., Pearson correlation) for all
    *numeric* columns in the specified dataset.

    Args:
        `dataset_id` (int): The unique identifier of the dataset.
        `db` (Session): **Dependency:** Database session managed by FastAPI.

    Raises:
        `HTTPException`:
            - `404` (Not Found): If a dataset with the specified `dataset_id` does not exist.
            - (Implicit via `load_csv_as_dataframe`): If the dataset file associated with
              the `dataset_id` cannot be found or loaded.

    Returns:
        `dict`: A dictionary representation of the pandas DataFrame containing the
                correlation matrix. NaN values in the matrix (e.g., if a column has zero variance)
                are replaced with `0`.
    """
    # 1. **Retrieve Dataset Record**: Fetch the dataset metadata from the database.
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    # 2. **Validate Dataset Existence**: Raise 404 if not found.
    if not dataset:
        # Typo in original code ('details='), corrected to 'detail='
        raise HTTPException(status_code=404, detail='Dataset not found')

    # 3. **Load Dataset Data**: Load the data from the CSV file.
    df = load_csv_as_dataframe(dataset.file_path)

    # 4. **Select Numeric Columns**: Filter the DataFrame to include only columns with numeric data types,
    #    as correlation is typically calculated only for these.
    numeric_df = df.select_dtypes(include='number')

    # 5. **Calculate Correlation Matrix**: Compute the pairwise correlation between numeric columns.
    #    The default method for pandas `.corr()` is usually Pearson.
    corr_matrix = numeric_df.corr()

    # 6. **Handle NaN Values**: Replace any NaN values in the resulting correlation matrix with 0.
    #    NaNs can occur if a column has constant values (zero standard deviation).
    corr_matrix_filled = corr_matrix.fillna(0)

    # 7. **Format and Return**: Convert the correlation matrix DataFrame to a dictionary for the JSON response.
    return corr_matrix_filled.to_dict()