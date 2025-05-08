import base64
from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Body
from typing import Optional, Annotated
import json # For parsing JSON string from form data

from app.services.ai_summary_service import ai_summary_service_instance
from app.schemas.ai_summary import (
    AISummaryResponse,
    SummaryStatsInput,
    CorrelationMatrixInput,
    ModelPerformanceInput,
    TunedModelInput,
    TunedModelResultsData
)

router = APIRouter(
    prefix='/v1',
    tags=['aisummary']
)

@router.post("/summary-stats", response_model=AISummaryResponse)
async def get_summary_statistics_analysis(
    payload: SummaryStatsInput
):
    try:
        summary = ai_summary_service_instance.get_ai_summary(
            data=payload.data,
            input_type='summary_stats'
        )
        if summary.startswith("Error:"):
            raise HTTPException(status_code=500, detail=summary)
        return AISummaryResponse(
            summary_html=summary,
            input_type='summary_stats'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/correlation-matrix", response_model=AISummaryResponse)
async def get_correlation_matrix_analysis(
    payload: CorrelationMatrixInput
):
    try:
        summary = ai_summary_service_instance.get_ai_summary(
            data=payload.data,
            input_type='correlation_matrix'
        )
        if summary.startswith("Error:"):
            raise HTTPException(status_code=500, detail=summary)
        return AISummaryResponse(
            summary_html=summary,
            input_type='correlation_matrix'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model-performance", response_model=AISummaryResponse)
async def get_model_performance_analysis(
    payload: ModelPerformanceInput
):
    try:
        summary = ai_summary_service_instance.get_ai_summary(
            data=payload.model_dump(), # Pass the whole dict
            input_type='model_performance'
        )
        if summary.startswith("Error:"):
            raise HTTPException(status_code=500, detail=summary)
        return AISummaryResponse(
            summary_html=summary,
            input_type='model_performance'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tuned-model-evaluation", response_model=AISummaryResponse)
async def get_tuned_model_evaluation(
    tuning_data_json: Annotated[str, Form()], # TunedModelResultsData as JSON string
    feature_importance_image: Annotated[Optional[UploadFile], File()] = None,
    image_url: Annotated[Optional[str], Form()] = None, # Allow providing a public URL too
):
    try:
        try:
            tuning_data_dict = json.loads(tuning_data_json)
            # Validate with Pydantic model
            validated_tuning_data = TunedModelResultsData(**tuning_data_dict)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for tuning_data_json.")
        except Exception as pydantic_error: # Catches Pydantic validation errors
            raise HTTPException(status_code=422, detail=f"Invalid tuning_data structure: {pydantic_error}")


        service_payload = {"tuning_data": validated_tuning_data.model_dump()}

        if feature_importance_image:
            if feature_importance_image.content_type not in ["image/png", "image/jpeg", "image/gif", "image/webp"]:
                raise HTTPException(status_code=400, detail="Invalid image file type. Supported: PNG, JPEG, GIF, WebP.")
            image_bytes = await feature_importance_image.read()
            service_payload["image_base64"] = base64.b64encode(image_bytes).decode('utf-8')
            service_payload["image_mime_type"] = feature_importance_image.content_type
        elif image_url:
            service_payload["image_url"] = image_url
        # If neither is provided, the service will handle it (prompt might mention missing image)

        summary = ai_summary_service_instance.get_ai_summary(
            data=service_payload,
            input_type='tuned_model_with_image_eval'
        )
        if summary.startswith("Error:"):
            raise HTTPException(status_code=500, detail=summary)
        return AISummaryResponse(
            summary_html=summary,
            input_type='tuned_model_with_image_eval'
        )
    except HTTPException: # Re-raise FastAPIs HTTPExceptions
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")