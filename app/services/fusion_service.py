import json
import numpy as np
from sqlalchemy.orm import Session

from app.db.models import InferenceResult
from app.core.class_labels import CLASS_LABELS


def build_fusion_result(mri_result: dict, eeg_result: dict):
    mri_probs = np.array(mri_result["probabilities"], dtype=np.float32)
    eeg_probs = np.array(eeg_result["probabilities"], dtype=np.float32)

    final_probs = (mri_probs * 0.6) + (eeg_probs * 0.4)

    final_index = int(final_probs.argmax())
    final_label = CLASS_LABELS.get(final_index, f"class_{final_index}")
    final_confidence = float(final_probs.max())

    return {
        "prediction_index": final_index,
        "prediction_label": final_label,
        "confidence": final_confidence,
        "probabilities": final_probs.tolist(),
        "fusion_method": "late_fusion_weighted_average",
        "fusion_method_label": "Late Fusion (Weighted Average)",
        "weights": {
            "mri": 0.6,
            "eeg": 0.4
        }
    }


def save_inference_result(
    db: Session,
    result: dict,
    mri_filename: str | None = None,
    heatmap_url: str | None = None,
    overlay_url: str | None = None,
    xai_method: str | None = None,
    explanation_text: str | None = None,
):
    mri_result = result["mri_result"]
    eeg_result = result["eeg_result"]
    fusion_result = result["fusion_result"]

    new_record = InferenceResult(
        mri_filename=mri_filename,

        mri_prediction_index=mri_result["prediction_index"],
        mri_prediction_label=mri_result["prediction_label"],
        mri_confidence=mri_result["confidence"],
        mri_probabilities=json.dumps(mri_result["probabilities"]),

        eeg_prediction_index=eeg_result["prediction_index"],
        eeg_prediction_label=eeg_result["prediction_label"],
        eeg_confidence=eeg_result["confidence"],
        eeg_probabilities=json.dumps(eeg_result["probabilities"]),

        fusion_prediction_index=fusion_result["prediction_index"],
        fusion_prediction_label=fusion_result["prediction_label"],
        fusion_confidence=fusion_result["confidence"],
        fusion_probabilities=json.dumps(fusion_result["probabilities"]),

        heatmap_url=heatmap_url,
        overlay_url=overlay_url,
        xai_method=xai_method,
        explanation_text=explanation_text,
    )

    db.add(new_record)
    db.commit()
    db.refresh(new_record)

    return new_record