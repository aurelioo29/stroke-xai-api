import numpy as np

from app.core.config import eeg_session
from app.core.class_labels import CLASS_LABELS
from app.utils.preprocess_eeg import preprocess_eeg_array
from app.utils.explanation import generate_eeg_xai_explanation


def run_eeg_inference(input_data: np.ndarray):
    input_name = eeg_session.get_inputs()[0].name
    output_name = eeg_session.get_outputs()[0].name

    outputs = eeg_session.run([output_name], {input_name: input_data})
    probs = outputs[0][0]

    prediction_index = int(np.argmax(probs))
    confidence = float(np.max(probs))

    return {
        "prediction_index": prediction_index,
        "prediction_label": CLASS_LABELS.get(
            prediction_index,
            f"class_{prediction_index}",
        ),
        "confidence": confidence,
        "probabilities": probs.tolist(),
    }


def get_importance_level(importance: float):
    safe_importance = max(float(importance), 0.0)

    if safe_importance >= 0.08:
        return {
            "level": "very_high",
            "color": "#ef4444",
            "label": "Kontribusi sangat tinggi",
            "description": "Area ini paling kuat memengaruhi prediksi model.",
        }

    if safe_importance >= 0.04:
        return {
            "level": "high",
            "color": "#f97316",
            "label": "Kontribusi tinggi",
            "description": "Area ini cukup kuat mendukung hasil prediksi.",
        }

    if safe_importance >= 0.015:
        return {
            "level": "medium",
            "color": "#eab308",
            "label": "Kontribusi sedang",
            "description": "Area ini memiliki pengaruh sedang terhadap prediksi.",
        }

    return {
        "level": "low",
        "color": "#3b82f6",
        "label": "Kontribusi rendah",
        "description": "Area ini berpengaruh kecil terhadap prediksi.",
    }


def get_color_legend():
    return [
        {
            "color": "#ef4444",
            "level": "Merah",
            "label": "Kontribusi sangat tinggi",
            "description": "Bagian sinyal yang paling kuat digunakan model untuk mendukung prediksi.",
        },
        {
            "color": "#f97316",
            "level": "Oranye",
            "label": "Kontribusi tinggi",
            "description": "Bagian sinyal yang cukup kuat memengaruhi hasil prediksi.",
        },
        {
            "color": "#eab308",
            "level": "Kuning",
            "label": "Kontribusi sedang",
            "description": "Bagian sinyal yang ikut berpengaruh, tetapi bukan faktor utama.",
        },
        {
            "color": "#3b82f6",
            "level": "Biru",
            "label": "Kontribusi rendah",
            "description": "Bagian sinyal dengan pengaruh kecil terhadap hasil prediksi.",
        },
    ]


def generate_eeg_occlusion_segments(
    eeg_array: list,
    target_class: int,
    segment_size: int = 150,
    stride: int = 75,
):
    base_input = preprocess_eeg_array(eeg_array)
    base_result = run_eeg_inference(base_input)
    base_confidence = float(base_result["probabilities"][target_class])

    signal = np.array(eeg_array, dtype=np.float32)

    important_segments = []

    for start in range(0, len(signal) - segment_size + 1, stride):
        end = start + segment_size

        occluded_signal = signal.copy()

        # Occlusion: bagian sinyal dibuat 0 untuk melihat pengaruhnya
        occluded_signal[start:end] = 0.0

        occluded_input = preprocess_eeg_array(occluded_signal.tolist())
        result = run_eeg_inference(occluded_input)
        occluded_confidence = float(result["probabilities"][target_class])

        raw_importance = base_confidence - occluded_confidence
        safe_importance = max(float(raw_importance), 0.0)

        importance_info = get_importance_level(safe_importance)

        important_segments.append({
            "start": start,
            "end": end,
            "importance": round(safe_importance, 6),
            "importance_percent": round(safe_importance * 100, 2),
            "raw_importance": round(float(raw_importance), 6),
            "base_confidence": round(base_confidence, 6),
            "occluded_confidence": round(occluded_confidence, 6),
            "level": importance_info["level"],
            "color": importance_info["color"],
            "label": importance_info["label"],
            "description": importance_info["description"],
        })

    important_segments.sort(
        key=lambda item: item["importance"],
        reverse=True,
    )

    return important_segments, base_result


async def predict_eeg_with_xai(eeg_data: list):
    input_data = preprocess_eeg_array(eeg_data)

    prediction_result = run_eeg_inference(input_data)
    target_class = prediction_result["prediction_index"]

    important_segments, _ = generate_eeg_occlusion_segments(
        eeg_array=eeg_data,
        target_class=target_class,
        segment_size=150,
        stride=75,
    )

    top_segments = important_segments[:5]

    explanation_text = generate_eeg_xai_explanation(
        prediction_label=prediction_result["prediction_label"],
        confidence=prediction_result["confidence"],
        important_segments=top_segments,
    )

    return {
        "prediction_index": prediction_result["prediction_index"],
        "prediction_label": prediction_result["prediction_label"],
        "confidence": prediction_result["confidence"],
        "probabilities": prediction_result["probabilities"],
        "xai_method": "occlusion_segment_sensitivity",
        "xai_method_label": "Occlusion Segment Sensitivity",
        "important_segments": top_segments,
        "color_legend": get_color_legend(),
        "explanation_text": explanation_text,
        "message": "EEG inference + XAI berhasil",
    }