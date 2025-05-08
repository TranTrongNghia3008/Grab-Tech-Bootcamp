from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Optional

# --- Schemas for input data structures ---

class SummaryStatItem(BaseModel):
    column: str
    count: int
    unique: Union[str, int, None] = None # Allow empty string or actual int
    top: Union[str, int, float, None] = None
    freq: Union[str, int, float, None] = None
    mean: Union[str, float, None] = None
    std: Union[str, float, None] = None
    min: Union[str, int, float, None] = None
    percent_25: Union[str, float, None] = Field(None, alias="25%")
    percent_50: Union[str, float, None] = Field(None, alias="50%")
    percent_75: Union[str, float, None] = Field(None, alias="75%")
    max: Union[str, int, float, None] = None

    class Config:
        populate_by_name = True # Allows using "25%" in JSON

class SummaryStatsInput(BaseModel):
    data: List[SummaryStatItem]

class CorrelationMatrixInput(BaseModel):
    data: Dict[str, Dict[str, float]]

class ModelPerformanceInput(BaseModel):
    columns: List[str]
    data: List[List[Union[str, float, int]]]

class TunedModelBestParams(BaseModel):
    copy_X: Optional[bool] = None
    fit_intercept: Optional[bool] = None
    n_jobs: Optional[int] = None
    positive: Optional[bool] = None

class TunedModelCVMetricsRow(BaseModel):
    fold_or_stat: Union[str, int] = Field(..., alias="Fold") # To match "Fold" or "Mean"/"Std"
    mae: float = Field(..., alias="MAE")
    mse: float = Field(..., alias="MSE")
    rmse: float = Field(..., alias="RMSE")
    r2: float = Field(..., alias="R2")
    rmsle: float = Field(..., alias="RMSLE")
    mape: float = Field(..., alias="MAPE")
    
    class Config:
        populate_by_name = True

class BaselineModelMetricsData(BaseModel):
    accuracy: float = Field(..., ge=0, le=1)
    auc: float = Field(..., ge=0, le=1) # Typically 0.5 to 1 for useful models
    recall: float = Field(..., ge=0, le=1)
    precision: float = Field(..., ge=0, le=1)
    f1: float = Field(..., ge=0, le=1, alias="F1") # Alias for F1-score
    kappa: float = Field(..., ge=-1, le=1) # Cohen's Kappa
    mcc: float = Field(..., ge=-1, le=1)   # Matthews Correlation Coefficient

    class Config:
        populate_by_name = True # To allow "F1" as input key for f1 field


class TunedModelCVMetricsTable(BaseModel):
    columns: List[str]
    data: List[List[Union[str, float, int]]] # Simpler for now, can be List[TunedModelCVMetricsRow] with more effort if strict typing for each cell is needed

class TunedModelResultsData(BaseModel):
    best_params: Dict[str, Any] # Using Dict[str, Any] for flexibility as in example
    cv_metrics_table: TunedModelCVMetricsTable

class TunedModelInput(BaseModel):
    tuning_data: TunedModelResultsData
    image_url: Optional[str] = None # For providing a public URL to an image

# --- Response Schema ---
class AISummaryResponse(BaseModel):
    summary_html: str
    input_type: str