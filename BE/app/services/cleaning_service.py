from app.utils.cleaning_tool import automate_cleaning_by_task
from app.crud.datasets import get_dataset_by_id, create_dataset
from app.crud.cleaning_jobs import (
    create_cleaning_job, get_cleaning_job_by_id, update_cleaning_job
)
from app.utils.file_storage import load_csv_as_dataframe
from app.db.session import SessionLocal
from sqlalchemy.orm import Session
import time
import logging # Import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def schedule_cleaning(dataset_id: int, config: dict, db) -> int:
    job = create_cleaning_job(db, dataset_id, config)
    return job.id


def run_cleaning_job(job_id: int):
    db: Session | None = None
    
    try:
        db = SessionLocal()
        job = get_cleaning_job_by_id(db, job_id)
        job = update_cleaning_job(db, job, status='running')

        ds = get_dataset_by_id(db, job.dataset_id)
        original = ds.file_path
        output = f"dataset_{ds.id}_cleaned_{job.id}.csv"

        automate_cleaning_by_task(
            input_csv=original,
            output_csv=output,
            remove_duplicates=job.config.get('remove_duplicates', False),
            handle_missing_values=job.config.get('handle_missing_values', False),
            smooth_noisy_data=job.config.get('smooth_noisy_data', False),
            handle_outliers=job.config.get('handle_outliers', False),
            reduce_cardinality=job.config.get('reduce_cardinality', False),
            encode_categorical_values=job.config.get('encode_categorical_values', False),
            feature_scaling=job.config.get('feature_scaling', False),
        )

        new_ds = create_dataset(db, ds.connection_id, output)
        df_orig = load_csv_as_dataframe(original)
        df_clean = load_csv_as_dataframe(output)

        results = {
            'original_rows': len(df_orig),
            'cleaned_rows': len(df_clean),
            'cleaned_dataset_id': new_ds.id
        }

        job = update_cleaning_job(db, job, status='completed', results=results)
    finally:
        if db:
            db.close()


def preview_cleaning(dataset_id: int, db) -> dict:
    from app.utils.file_storage import load_csv_as_dataframe
    ds = get_dataset_by_id(db, dataset_id)
    df = load_csv_as_dataframe(ds.file_path)
    missing = df.isna().sum().to_dict()
    outliers = {col: int(((df[col]-df[col].mean()).abs()>3*df[col].std()).sum())
                for col in df.select_dtypes(include='number')}
    duplicates = int(df.duplicated().sum())
    return {'missing': missing, 'outliers': outliers, 'duplicates': duplicates}


def get_status(job_id: int, db) -> str:
    return get_cleaning_job_by_id(db, job_id).status


def get_results(job_id: int, db) -> dict:
    job = get_cleaning_job_by_id(db, job_id)
    return job.results


def update_cleaning(job_id: int, config: dict, db) -> bool:
    job = get_cleaning_job_by_id(db, job_id)
    if not job: return False
    update_cleaning_job(db, job, config=config)
    return True


def delete_cleaning(job_id: int, db) -> bool:
    from app.crud.cleaning_jobs import delete_cleaning_job
    return delete_cleaning_job(db, job_id)