import json

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Body,
    Form,
    Depends,
    Query,
    HTTPException,
)
from sqlalchemy.orm import Session

from app.services.mri_service import predict_mri
from app.services.eeg_service import predict_eeg
from app.services.mri_xai_service import predict_mri_with_xai
from app.services.eeg_xai_service import predict_eeg_with_xai
from app.services.fusion_service import build_fusion_result, save_inference_result
from app.core.config import get_model_io_details, mri_session, eeg_session
from app.db.database import get_db
from app.db.models import InferenceResult
from app.utils.explanation import generate_multimodal_explanation
from app.utils.eeg_csv import read_eeg_csv

try:
    from app.services.llm_explanation_service import (
        generate_llm_multimodal_explanation,
    )
except Exception:
    generate_llm_multimodal_explanation = None


router = APIRouter(prefix="/predict", tags=["Prediction"])


# ============================================================
# Helpers
# ============================================================

def safe_parse_json_array(value):
    if value is None:
        return []

    if isinstance(value, list):
        return value

    try:
        return json.loads(value)
    except Exception:
        return []


def safe_get_attr(item, attr_name, default=None):
    """
    Supaya aman kalau column baru belum ada di database model.
    Jadi history tidak langsung error.
    """
    return getattr(item, attr_name, default)


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

        # Existing fields
        "heatmap_url": item.heatmap_url,
        "overlay_url": item.overlay_url,
        "xai_method": item.xai_method,
        "explanation_text": item.explanation_text,

        # Optional newer fields kalau nanti kamu tambah column DB
        "raw_heatmap_url": safe_get_attr(item, "raw_heatmap_url"),
        "disease_focus_url": safe_get_attr(item, "disease_focus_url"),
        "zone_analysis": safe_parse_json_array(
            safe_get_attr(item, "zone_analysis")
        ),
        "clinical_message": safe_get_attr(item, "clinical_message"),

        "created_at": item.created_at,
    }


def build_mri_result_payload(mri_xai_result: dict):
    """
    Payload MRI yang sudah membawa zone-based XAI.
    Ini dipakai untuk response MRI-only dan multimodal.
    """

    return {
        "prediction_index": mri_xai_result.get("prediction_index"),
        "prediction_label": mri_xai_result.get("prediction_label"),
        "confidence": mri_xai_result.get("confidence"),
        "probabilities": mri_xai_result.get("probabilities", []),

        # XAI URLs
        "raw_heatmap_url": mri_xai_result.get("raw_heatmap_url"),
        "heatmap_url": mri_xai_result.get("raw_heatmap_url"),
        "overlay_url": mri_xai_result.get("overlay_url"),
        "disease_focus_url": mri_xai_result.get("disease_focus_url"),

        # Zone-based clinical data
        "zone_analysis": mri_xai_result.get("zone_analysis"),
        "clinical_message": mri_xai_result.get("clinical_message"),

        # Notes
        "heatmap_legend": mri_xai_result.get("heatmap_legend"),
        "clinical_note": mri_xai_result.get("clinical_note"),
        "xai_method": mri_xai_result.get("xai_method"),

        # Explanation
        "explanation_text": mri_xai_result.get("explanation_text"),
        "explanation_source": mri_xai_result.get("explanation_source"),

        "message": mri_xai_result.get(
            "message",
            "MRI inference + zone-based XAI berhasil",
        ),
    }


# ============================================================
# Model Inspection
# ============================================================

@router.get("/inspect-models")
async def inspect_models():
    return {
        "success": True,
        "data": {
            "mri": get_model_io_details(mri_session),
            "eeg": get_model_io_details(eeg_session),
        },
    }


# ============================================================
# MRI Prediction
# ============================================================

@router.post("/mri")
async def predict_mri_route(file: UploadFile = File(...)):
    result = await predict_mri(file)

    return {
        "success": True,
        "data": result,
    }


@router.post("/mri-xai")
async def predict_mri_xai_route(file: UploadFile = File(...)):
    """
    MRI XAI dengan zone-based clinical overlay.

    Untuk normal:
    - overlay_url tidak menampilkan area merah.
    - zone_analysis.is_disease_prediction = False.

    Untuk hemorrhagic / ischemic:
    - overlay_url menampilkan disease focus zone.
    - disease_focus_url juga tersedia.
    """

    result = await predict_mri_with_xai(file)

    return {
        "success": True,
        "data": build_mri_result_payload(result),
    }


# ============================================================
# EEG Prediction
# ============================================================

@router.post("/eeg")
async def predict_eeg_route(eeg_array: list = Body(...)):
    result = await predict_eeg(eeg_array)

    return {
        "success": True,
        "data": result,
    }


@router.post("/eeg-xai")
async def predict_eeg_xai_route(eeg_array: list = Body(...)):
    result = await predict_eeg_with_xai(eeg_array)

    return {
        "success": True,
        "data": result,
    }


@router.post("/eeg-xai-csv")
async def predict_eeg_xai_csv_route(
    file: UploadFile = File(...),
    row_index: int = Form(0),
    graph_channel: int = Form(1),

    # P1-P4
    section_count: int = Form(4),

    # 20 sample per section:
    # P1 = s1-s20
    # P2 = s21-s40
    # P3 = s41-s60
    # P4 = s61-s80
    section_size: int = Form(20),

    # 2 cycle:
    # C1 = P1-P4
    # C2 = P1-P4
    #
    # isi 0 kalau mau auto sampai sample habis
    cycle_count: int = Form(2),
):
    csv_result = await read_eeg_csv(
        file=file,
        row_index=row_index,
        graph_channel=graph_channel,
        section_count=section_count,
        section_size=section_size,
        cycle_count=cycle_count,
    )

    result = await predict_eeg_with_xai(
        eeg_data=csv_result["model_input"],
        graph_sections=csv_result["graph_sections"],
    )

    return {
        "success": True,
        "data": {
            **result,
            "graph_data": csv_result["graph_data"],
            "graph_sections": csv_result["graph_sections"],
            "feature_count": csv_result["feature_count"],
            "selected_row": csv_result["selected_row"],
            "selected_channel": csv_result["selected_channel"],
            "section_count": csv_result["section_count"],
            "section_size": csv_result["section_size"],
            "cycle_count": csv_result["cycle_count"],
            "feature_selection_info": csv_result["feature_selection_info"],
            "uploaded_filename": csv_result["uploaded_filename"],
        },
    }


# ============================================================
# Multimodal Prediction
# ============================================================

@router.post("/multimodal")
async def predict_multimodal_route(
    file: UploadFile = File(...),
    eeg_json: str = Form(...),
    db: Session = Depends(get_db),
):
    """
    Fusion MRI + EEG.

    MRI menggunakan zone-based XAI.
    EEG menggunakan inference biasa.
    Fusion menggabungkan probability MRI + EEG.
    """

    try:
        eeg_array = json.loads(eeg_json)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Format eeg_json tidak valid. Harus JSON array.",
        )

    # 1. MRI dengan zone-based XAI
    mri_xai_result = await predict_mri_with_xai(file)
    mri_payload = build_mri_result_payload(mri_xai_result)

    # 2. EEG inference
    eeg_result = await predict_eeg(eeg_array)

    # 3. Fusion
    fusion_result = build_fusion_result(
        mri_result={
            "prediction_index": mri_xai_result["prediction_index"],
            "prediction_label": mri_xai_result["prediction_label"],
            "confidence": mri_xai_result["confidence"],
            "probabilities": mri_xai_result["probabilities"],
        },
        eeg_result=eeg_result,
    )

    # 4. Default explanation: rule-based
    explanation_text = generate_multimodal_explanation(
        mri_label=mri_xai_result["prediction_label"],
        eeg_label=eeg_result["prediction_label"],
        final_label=fusion_result["prediction_label"],
        confidence=fusion_result["confidence"],
    )
    explanation_source = "rule_based"

    # 5. Optional LLM explanation
    if generate_llm_multimodal_explanation is not None:
        try:
            explanation_text = generate_llm_multimodal_explanation(
                mri_label=mri_xai_result["prediction_label"],
                eeg_label=eeg_result["prediction_label"],
                final_label=fusion_result["prediction_label"],
                confidence=fusion_result["confidence"],
                xai_method=mri_xai_result.get(
                    "xai_method",
                    "zone_based_occlusion_sensitivity",
                ),
            )
            explanation_source = "llm"
        except Exception as e:
            print("LLM multimodal explanation error:", e)

    # 6. Final response
    result = {
        "mri_result": {
            "prediction_index": mri_payload["prediction_index"],
            "prediction_label": mri_payload["prediction_label"],
            "confidence": mri_payload["confidence"],
            "probabilities": mri_payload["probabilities"],

            "raw_heatmap_url": mri_payload.get("raw_heatmap_url"),
            "heatmap_url": mri_payload.get("heatmap_url"),
            "overlay_url": mri_payload.get("overlay_url"),
            "disease_focus_url": mri_payload.get("disease_focus_url"),

            "zone_analysis": mri_payload.get("zone_analysis"),
            "clinical_message": mri_payload.get("clinical_message"),
            "heatmap_legend": mri_payload.get("heatmap_legend"),
            "clinical_note": mri_payload.get("clinical_note"),

            "xai_method": mri_payload.get("xai_method"),
            "message": "MRI inference berhasil",
        },

        "eeg_result": eeg_result,

        "fusion_result": fusion_result,

        "xai_result": {
            "raw_heatmap_url": mri_payload.get("raw_heatmap_url"),
            "heatmap_url": mri_payload.get("heatmap_url"),
            "overlay_url": mri_payload.get("overlay_url"),
            "disease_focus_url": mri_payload.get("disease_focus_url"),
            "zone_analysis": mri_payload.get("zone_analysis"),
            "clinical_message": mri_payload.get("clinical_message"),
            "xai_method": mri_payload.get("xai_method"),
        },

        "explanation_text": explanation_text,
        "explanation_source": explanation_source,

        "message": "Fusion MRI dan EEG berhasil",
    }

    # 7. Save history
    #
    # Tetap kirim heatmap_url dan overlay_url field lama,
    # supaya save_inference_result tidak perlu langsung kamu rombak.
    #
    # heatmap_url pakai raw_heatmap_url.
    # overlay_url pakai clinical overlay.
    saved = save_inference_result(
        db=db,
        result=result,
        mri_filename=file.filename,
        heatmap_url=mri_payload.get("raw_heatmap_url"),
        overlay_url=mri_payload.get("overlay_url"),
        xai_method=mri_payload.get(
            "xai_method",
            "zone_based_occlusion_sensitivity",
        ),
        explanation_text=explanation_text,
    )

    return {
        "success": True,
        "data": result,
        "saved_result_id": saved.id,
    }


# ============================================================
# History
# ============================================================

@router.get("/history")
def get_inference_history(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
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
            "total": total,
        },
        "data": [format_inference_result(item) for item in results],
    }


@router.get("/history/{result_id}")
def get_inference_history_detail(
    result_id: int,
    db: Session = Depends(get_db),
):
    item = (
        db.query(InferenceResult)
        .filter(InferenceResult.id == result_id)
        .first()
    )

    if not item:
        raise HTTPException(
            status_code=404,
            detail="Data history tidak ditemukan",
        )

    return {
        "success": True,
        "data": format_inference_result(item),
    }