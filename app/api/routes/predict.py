import json
import numpy as np

from fastapi import APIRouter, UploadFile, File, Body, Form, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from app.services.mri_service import predict_mri
from app.services.eeg_service import predict_eeg
from app.services.mri_xai_service import predict_mri_with_xai
from app.services.fusion_service import build_fusion_result, save_inference_result
from app.core.config import get_model_io_details, mri_session, eeg_session
from app.db.database import get_db
from app.db.models import InferenceResult
from app.utils.explanation import generate_multimodal_explanation
from app.services.eeg_xai_service import predict_eeg_with_xai

router = APIRouter(prefix="/predict", tags=["Prediction"])


def safe_parse_json_array(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except Exception:
        return []


def format_inference_result(item: InferenceResult):
    return {
        "id": item.id,
        "mri_filename": item.mri_filename,

        "mri_prediction_index": item.mri_prediction_index,
        "mri_prediction_label": item.mri_prediction_label,
        "mri_confidence": item.mri_confidence,
        "mri_probabilities": safe_parse_json_array(item.mri_probabilities),

        "eeg_prediction_index": item.eeg_prediction_index,
        "eeg_prediction_label": item.eeg_prediction_label,
        "eeg_confidence": item.eeg_confidence,
        "eeg_probabilities": safe_parse_json_array(item.eeg_probabilities),

        "fusion_prediction_index": item.fusion_prediction_index,
        "fusion_prediction_label": item.fusion_prediction_label,
        "fusion_confidence": item.fusion_confidence,
        "fusion_probabilities": safe_parse_json_array(item.fusion_probabilities),

        "heatmap_url": item.heatmap_url,
        "overlay_url": item.overlay_url,
        "xai_method": item.xai_method,
        "explanation_text": item.explanation_text,

        "created_at": item.created_at,
    }


@router.get("/inspect-models")
async def inspect_models():
    return {
        "success": True,
        "data": {
            "mri": get_model_io_details(mri_session),
            "eeg": get_model_io_details(eeg_session),
        }
    }


@router.post("/mri")
async def predict_mri_route(file: UploadFile = File(...)):
    result = await predict_mri(file)
    return {
        "success": True,
        "data": result
    }


@router.post("/mri-xai")
async def predict_mri_xai_route(file: UploadFile = File(...)):
    result = await predict_mri_with_xai(file)
    return {
        "success": True,
        "data": result
    }


@router.post("/eeg")
async def predict_eeg_route(eeg_array: list = Body(...)):
    result = await predict_eeg(eeg_array)
    return {
        "success": True,
        "data": result
    }


@router.post("/multimodal")
async def predict_multimodal_route(
    file: UploadFile = File(...),
    eeg_json: str = Form(...),
    db: Session = Depends(get_db)
):
    eeg_array = json.loads(eeg_json)

    # MRI dengan XAI
    mri_xai_result = await predict_mri_with_xai(file)

    # EEG inference
    eeg_result = await predict_eeg(eeg_array)

    # Fusion
    fusion_result = build_fusion_result(
        mri_result={
            "prediction_index": mri_xai_result["prediction_index"],
            "prediction_label": mri_xai_result["prediction_label"],
            "confidence": mri_xai_result["confidence"],
            "probabilities": mri_xai_result["probabilities"],
        },
        eeg_result=eeg_result
    )

    explanation_text = generate_multimodal_explanation(
        mri_label=mri_xai_result["prediction_label"],
        eeg_label=eeg_result["prediction_label"],
        final_label=fusion_result["prediction_label"],
        confidence=fusion_result["confidence"],
    )

    result = {
        "mri_result": {
            "prediction_index": mri_xai_result["prediction_index"],
            "prediction_label": mri_xai_result["prediction_label"],
            "confidence": mri_xai_result["confidence"],
            "probabilities": mri_xai_result["probabilities"],
            "message": "MRI inference berhasil"
        },
        "eeg_result": eeg_result,
        "fusion_result": fusion_result,
        "xai_result": {
            "heatmap_url": mri_xai_result["heatmap_url"],
            "overlay_url": mri_xai_result["overlay_url"],
            "xai_method": mri_xai_result["xai_method"]
        },
        "explanation_text": explanation_text,
        "message": "Fusion MRI dan EEG berhasil"
    }

    saved = save_inference_result(
        db=db,
        result=result,
        mri_filename=file.filename,
        heatmap_url=mri_xai_result["heatmap_url"],
        overlay_url=mri_xai_result["overlay_url"],
        xai_method=mri_xai_result["xai_method"],
        explanation_text=explanation_text,
    )

    return {
        "success": True,
        "data": result,
        "saved_result_id": saved.id
    }


@router.get("/history")
def get_inference_history(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    results = (
        db.query(InferenceResult)
        .order_by(InferenceResult.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    total = db.query(InferenceResult).count()

    return {
        "success": True,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": total
        },
        "data": [format_inference_result(item) for item in results]
    }


@router.get("/history/{result_id}")
def get_inference_history_detail(
    result_id: int,
    db: Session = Depends(get_db)
):
    item = db.query(InferenceResult).filter(InferenceResult.id == result_id).first()

    if not item:
        raise HTTPException(status_code=404, detail="Data history tidak ditemukan")

    return {
        "success": True,
        "data": format_inference_result(item)
    }

@router.post("/eeg-xai")
async def predict_eeg_xai_route(eeg_array: list = Body(...)):
    result = await predict_eeg_with_xai(eeg_array)
    return {
        "success": True,
        "data": result
    }