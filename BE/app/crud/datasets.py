from sqlalchemy.orm import Session
from typing import Optional, List, Tuple
from pydantic import BaseModel
from datetime import datetime
import pandas as pd

from app.db.models.datasets import Dataset
from app.crud.base import CRUDBase
from app.schemas.dataset import DatasetInfo, DatasetAnalysisReport, FeatureProfile
from app.utils.file_storage import get_file_path, load_csv_as_dataframe
from ydata_profiling import ProfileReport


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
    
    def get_dataset_analysis(self, db: Session, dataset_id: int) -> DatasetAnalysisReport:
        """
        Analyzes a dataset and returns a comprehensive report.
        """
        db_dataset = self.get(db=db, id=dataset_id) # Assuming self.get fetches a Dataset instance
        if not db_dataset:
            return None # Or raise an exception to be caught by the endpoint

        if not db_dataset.file_path:
            # Handle case where file_path might not be set (e.g., during creation error)
            # For this analysis, we need the file path.
            raise ValueError(f"Dataset ID {dataset_id} does not have a valid file_path.")

        # Load dataframe - assuming cleaned data is preferred if available
        # Modify this logic if you want to analyze the original or a specific version
        file_to_analyze = db_dataset.file_path # Default to the main file_path
        
        # Construct the path to the cleaned file if it's a convention
        # For example, if cleaned file is always dataset_{id}_cleaned.csv
        cleaned_filename_candidate = f"dataset_{db_dataset.id}_cleaned.csv"
        # This is a bit of a guess; adapt to how you store cleaned file paths
        # Or, if you have a specific column for cleaned_file_path in Dataset, use that.
        try:
            # Try loading the cleaned file first if it's part of your workflow
            # This assumes get_file_path can check existence or load_csv_as_dataframe handles FileNotFoundError
            potential_cleaned_path = get_file_path(cleaned_filename_candidate)
            df = load_csv_as_dataframe(potential_cleaned_path)
            actual_file_path_used = potential_cleaned_path
            print(f"Analyzing cleaned dataset: {actual_file_path_used}")
        except FileNotFoundError:
            print(f"Cleaned dataset for ID {dataset_id} not found, analyzing original: {db_dataset.file_path}")
            df = load_csv_as_dataframe(db_dataset.file_path) # Fallback to original
            actual_file_path_used = db_dataset.file_path
        except Exception as e:
            raise RuntimeError(f"Error loading dataset {dataset_id} from {db_dataset.file_path}: {e}")


        if df.empty:
            # Handle empty dataframe case gracefully
            return DatasetAnalysisReport(
                dataset_id=db_dataset.id,
                project_name=db_dataset.project_name,
                file_path=actual_file_path_used,
                total_records=0,
                total_features=0,
                overall_missing_percentage=0.0,
                data_quality_score=0.0, # Or some other representation for empty
                features=[]
            )

        total_records = len(df)
        total_features = len(df.columns)

        # Overall missing percentage
        total_cells = df.size
        total_missing_cells = df.isnull().sum().sum()
        overall_missing_percentage = (total_missing_cells / total_cells) * 100 if total_cells > 0 else 0.0

        # Data Quality Score (simple definition for now)
        # 100 - overall_missing_percentage. Can be made more complex.
        # Lower missing values = higher score.
        data_quality_score = max(0.0, 100.0 - overall_missing_percentage)

        feature_profiles: List[FeatureProfile] = []
        for col_name in df.columns:
            series = df[col_name]
            missing_count = series.isnull().sum()
            missing_percentage = (missing_count / total_records) * 100 if total_records > 0 else 0.0

            # Unique percentage (count unique non-null values / total records)
            # nunique() counts distinct observations over requested axis (excluding NA by default)
            # series.count() gives non-NA count
            unique_count = series.nunique()
            unique_percentage = (unique_count / total_records) * 100 if total_records > 0 else 0.0
            # Alternative for unique_percentage: (series.nunique() / series.count()) * 100 if series.count() > 0 else 0.0
            # This would be "percentage of unique values among non-missing values".
            # The current one is "percentage of unique values across all records".

            dtype_str = map_dtype_to_simplified_type(series.dtype, series)

            feature_profiles.append(
                FeatureProfile(
                    feature_name=str(col_name), # Ensure column name is string
                    dtype=dtype_str,
                    missing_percentage=round(missing_percentage, 2),
                    unique_percentage=round(unique_percentage, 2)
                )
            )

        return DatasetAnalysisReport(
            dataset_id=db_dataset.id,
            project_name=db_dataset.project_name,
            file_path=actual_file_path_used, # The actual file analyzed
            total_records=total_records,
            total_features=total_features,
            overall_missing_percentage=round(overall_missing_percentage, 2),
            data_quality_score=round(data_quality_score, 2), # Or a more complex dict/object
            features=feature_profiles
        )
        
    def update_project_name(
        self,
        db: Session,
        *,
        dataset_id: int,
        new_project_name: Optional[str]
    ) -> Tuple[Optional[Dataset], Optional[str]]:
        """
        Updates the project_name of a specific dataset.

        Checks for uniqueness if the new name is not None.

        Returns:
            Tuple[Optional[Dataset], Optional[str]]: A tuple containing:
                - The updated Dataset object if successful and found.
                - An error message string if validation fails (e.g., uniqueness), None otherwise.
                  Returns (None, None) if the dataset_id itself was not found.
        """
        db_obj = self.get(db=db, id=dataset_id)
        if not db_obj:
            return None, None # Dataset not found

        # Check for uniqueness constraint if new name is provided and different
        if new_project_name is not None and new_project_name != db_obj.project_name:
            existing = db.query(Dataset).filter(
                Dataset.project_name == new_project_name,
                Dataset.id != dataset_id # Exclude the current dataset from the check
            ).first()
            if existing:
                error_msg = f"Project name '{new_project_name}' already exists for dataset ID {existing.id}."
                return db_obj, error_msg # Return current object and error message

        # Update the project name
        db_obj.project_name = new_project_name
        try:
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            return db_obj, None # Success
        except Exception as e:
            db.rollback()
            error_msg = f"Database error during update: {e}"
            return db_obj, error_msg

# --- Instantiate the CRUD class ---
crud_dataset = CRUDDataset(Dataset)