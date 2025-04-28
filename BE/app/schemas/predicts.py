from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any, Union
from app.schemas.commons import DataFrameStructure
from app.schemas.automl_jobs import AutoMLJobResultBase

class PredictionRequest(BaseModel):
    finalized_model_id: Optional[int] = None
    model_uri: Optional[str] = None
    new_data_path: Optional[str] = None
    new_data_records: Optional[List[Dict[str, Any]]] = None
    output_predictions_path: Optional[str] = None
    include_input_data: bool = Field(True)

    @model_validator(pre=True)
    def check_model_source(cls, values):
        fm_id, uri = values.get('finalized_model_id'), values.get('model_uri')
        if not (fm_id or uri) or (fm_id and uri): 
            raise ValueError("Provide exactly one of 'finalized_model_id' or 'model_uri'")
        return values

    @model_validator(pre=True)
    def check_data_source(cls, values):
        path, records = values.get('new_data_path'), values.get('new_data_records')
        if not (path or records) or (path and records): 
            raise ValueError("Provide exactly one of 'new_data_path' or 'new_data_records'")
        return values

class PredictionResult(AutoMLJobResultBase):
    predictions_output_path: Optional[str] = None
    predictions: Optional[Union[List[Dict[str, Any]], DataFrameStructure]] = None
    num_predictions: Optional[int] = None
    prediction_columns_added: Optional[List[str]] = None