from sqlalchemy.orm import Session
from typing import Optional, List, Tuple
from pydantic import BaseModel
from datetime import datetime
import pandas as pd

from app.db.models.datasets import Dataset
from app.crud.base import CRUDBase
from app.schemas.dataset import DatasetInfo, DatasetAnalysisReport, FeatureProfile
from app.utils.file_storage import get_file_path, load_csv_as_dataframe, load_csv_as_dask_dataframe, save_dask_dataframe_as_csv
from ydata_profiling import ProfileReport
import dask.dataframe as dd
import asyncio
import dask


def map_dtype_to_simplified_type(dtype_obj, series: pd.Series) -> str:
    """
    Maps a pandas dtype object to a simplified type string.
    For object types, it checks if it can be inferred as datetime.
    """
    if pd.api.types.is_integer_dtype(dtype_obj):
        return "integer"
    elif pd.api.types.is_float_dtype(dtype_obj):
        return "float"
    elif pd.api.types.is_datetime64_any_dtype(dtype_obj):
        return "datetime"
    elif pd.api.types.is_bool_dtype(dtype_obj):
        return "boolean"
    elif pd.api.types.is_object_dtype(dtype_obj):
        # Try to infer if object column is actually datetime-like
        try:
            # Attempt to convert a small sample to datetime to check
            # Be careful with large datasets, sample or use a more robust check
            if len(series) > 0:
                pd.to_datetime(series.dropna().sample(min(5, len(series.dropna()))), errors='raise')
                return "datetime" # If conversion works for a sample, assume datetime
            return "categorical" # Default for object
        except (ValueError, TypeError):
            return "categorical" # If conversion fails, it's likely string/categorical
    else:
        return str(dtype_obj) # Fallback to the original dtype string

# --- Schemas ---
# Define data structures for creating and updating Dataset records via the API/CRUD operations.
class DatasetCreateSchema(BaseModel):
    project_name: str
    file_path: str = ""
    connection_id: Optional[int] = None

class DatasetUpdateSchema(BaseModel):
    file_path: Optional[str] = None
    connection_id: Optional[int] = None

# --- CRUD Class using CRUDBase ---
class CRUDDataset(CRUDBase[Dataset, DatasetCreateSchema, DatasetUpdateSchema]):
    def get_all_datasets_ordered_by_creation(
        self,
        db: Session,
        descending: bool = True # True for newest first, False for oldest first
    ) -> List[Dataset]: # Returns a list of SQLAlchemy Dataset instances
        """
        Retrieves all dataset records, ordered by their creation date.
        """
        query = db.query(Dataset)

        if descending:
            # Order by created_at descending (newest first), then by ID as a tie-breaker
            query = query.order_by(Dataset.created_at.desc(), Dataset.id.desc())
        else:
            # Order by created_at ascending (oldest first), then by ID as a tie-breaker
            query = query.order_by(Dataset.created_at.asc(), Dataset.id.asc())

        return query.all() # Fetch all matching records
    
    async def get_dataset_analysis_dask(self, db: Session, dataset_id: int) -> Optional[DatasetAnalysisReport]:
        dataset_model = await asyncio.to_thread(db.query(Dataset).filter(Dataset.id == dataset_id).first)
        if not dataset_model:
            return None
        if not dataset_model.file_path:  # Or however you store the original path
            raise ValueError("Dataset has no file path")

        # Determine which file to load (cleaned or original)
        # Ensure this logic correctly points to an existing file
        file_to_analyze = get_file_path(f'dataset_{dataset_id}_cleaned.csv') # Your logic

        try:
            # OPTIMIZATION: Pass dtypes to read_csv if known!
            # e.g., dtypes = {'col1': 'int', 'col2': 'str'}
            # ddf = await asyncio.to_thread(load_csv_as_dask_dataframe, file_to_analyze, dtypes=your_dtypes_dict)
            ddf = await asyncio.to_thread(load_csv_as_dask_dataframe, file_to_analyze)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file for analysis not found at: {file_to_analyze}")
        except RuntimeError as e:
            raise RuntimeError(f"Error loading dataset for analysis: {e}")
        except Exception as e: # Catch broader exceptions during load
            raise RuntimeError(f"Unexpected error loading dataset: {e}")


        # Check if ddf is essentially empty (metadata checks, cheap)
        if ddf.npartitions == 0 or len(ddf.columns) == 0:
            return DatasetAnalysisReport(
                total_records=0, total_features=len(ddf.columns), overall_missing_percentage=0,
                data_quality_score=0, features=[]
            )

        # --- OPTIMIZATION: Define all Dask operations before computing ---
        # 1. Total records (as a delayed object, not computed yet)
        total_records_delayed = ddf.shape[0] # This is a Dask scalar, will compute to an int

        # 2. Missing counts per column (as a delayed Dask Series)
        missing_counts_delayed = ddf.isnull().sum() # This is a Dask Series

        # 3. Unique counts per column (as a dictionary of delayed Dask Scalars)
        # Note: dask.compute can take a dictionary of tasks and will return a dictionary of results
        unique_counts_delayed = {col: ddf[col].nunique_approx() for col in ddf.columns}

        # --- Execute all Dask computations in parallel with one call ---
        # dask.compute can take multiple arguments and will return a tuple of results
        # in the same order. If a dict is passed, a dict of results is returned.
        computed_values = await asyncio.to_thread(
            dask.compute,
            total_records_delayed,
            missing_counts_delayed,
            unique_counts_delayed
            # You can add more delayed objects here if needed
        )

        # Unpack the results
        total_records = computed_values[0]
        missing_counts_series = computed_values[1] # This is now a Pandas Series
        unique_counts_map = computed_values[2]     # This is now a dict of {col_name: unique_count}

        total_features = len(ddf.columns)

        # Handle case where dataset loaded but has zero records
        if total_records == 0:
            return DatasetAnalysisReport(
                total_records=0, total_features=total_features, overall_missing_percentage=0,
                data_quality_score=0, # Or some other appropriate score
                features=[
                    FeatureProfile(feature_name=str(col), dtype=str(ddf[col].dtype), missing_percentage=0, unique_percentage=0)
                    for col in ddf.columns
                ]
            )

        # Now perform calculations on the computed (Pandas/Python native) results
        total_missing = missing_counts_series.sum() # Operates on Pandas Series, fast
        overall_missing_percentage = (total_missing / (total_records * total_features)) * 100 if total_records > 0 and total_features > 0 else 0

        features_analysis: List[FeatureProfile] = []
        for col in ddf.columns:
            missing_val_for_col = missing_counts_series[col]
            missing_pct = (missing_val_for_col / total_records) * 100 if total_records > 0 else 0

            unique_count_for_col = unique_counts_map[col]
            unique_pct = (unique_count_for_col / total_records) * 100 if total_records > 0 else 0

            features_analysis.append(
                FeatureProfile(
                    feature_name=str(col),
                    dtype=str(ddf[col].dtype), # ddf[col].dtype is metadata, cheap
                    missing_percentage=round(missing_pct, 2),
                    unique_percentage=round(unique_pct, 2)
                )
            )

        data_quality_score = round(100 - overall_missing_percentage, 2)
        actual_file_path_used_for_report = dataset_model.file_path
        project_name_to_report = dataset_model.project_name if dataset_model.project_name is not None else "N/A"
        
        return DatasetAnalysisReport(
            dataset_id=dataset_model.id,
            project_name=project_name_to_report,
            file_path=actual_file_path_used_for_report,
            total_records=total_records,
            total_features=total_features,
            overall_missing_percentage=round(overall_missing_percentage, 2),
            data_quality_score=data_quality_score,
            features=features_analysis
        )

# --- Instantiate the CRUD class ---
crud_dataset = CRUDDataset(Dataset)