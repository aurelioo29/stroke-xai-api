import numpy as np

from app.core.config import eeg_session
from app.core.class_labels import CLASS_LABELS
from app.utils.preprocess_eeg import preprocess_eeg_array
from app.utils.explanation import generate_eeg_xai_explanation


# ============================================================
# Inference
# ============================================================

def run_eeg_inference(input_data: np.ndarray):
    input_name = eeg_session.get_inputs()[0].name
    output_name = eeg_session.get_outputs()[0].name

    outputs = eeg_session.run([output_name], {input_name: input_data})
    probs = outputs[0][0]
    probs = np.asarray(probs, dtype=np.float32)

    # Safety kalau output masih logits
    if (
        probs.min() < 0
        or probs.max() > 1
        or not np.isclose(probs.sum(), 1.0, atol=1e-3)
    ):
        exp_probs = np.exp(probs - np.max(probs))
        probs = exp_probs / exp_probs.sum()

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


# ============================================================
# Color Concept
# ============================================================

def get_color_level_from_percent(percent: float):
    """
    Konsep warna sesuai arahan:
    Biru  = Normal
    Hijau = Observasi normal
    Kuning = Early menuju prediksi penyakit
    Merah = Prediksi kuat penyakit
    """

    value = float(percent)

    if value >= 75:
        return {
            "level": "strong_disease",
            "color": "#ef4444",
            "label": "Prediksi kuat penyakit",
            "description": "Section ini memiliki kontribusi paling kuat terhadap prediksi model.",
        }

    if value >= 50:
        return {
            "level": "early_warning",
            "color": "#facc15",
            "label": "Early warning",
            "description": "Section ini mulai mengarah ke pola yang mendukung prediksi penyakit.",
        }

    if value >= 25:
        return {
            "level": "normal_observation",
            "color": "#22c55e",
            "label": "Observasi normal",
            "description": "Section ini masih relatif rendah, tetapi perlu diamati.",
        }

    return {
        "level": "normal",
        "color": "#2563eb",
        "label": "Normal",
        "description": "Section ini memiliki kontribusi rendah terhadap prediksi penyakit.",
    }


def get_color_legend():
    return [
        {
            "color": "#2563eb",
            "level": "Biru",
            "label": "Normal",
            "description": "Area sinyal dengan kontribusi rendah terhadap prediksi penyakit.",
        },
        {
            "color": "#22c55e",
            "level": "Hijau",
            "label": "Observasi normal",
            "description": "Area sinyal masih relatif rendah, tetapi mulai diamati.",
        },
        {
            "color": "#facc15",
            "level": "Kuning",
            "label": "Early warning",
            "description": "Area sinyal mulai mengarah ke pola pendukung prediksi penyakit.",
        },
        {
            "color": "#ef4444",
            "level": "Merah",
            "label": "Prediksi kuat penyakit",
            "description": "Area sinyal paling kuat mendukung prediksi penyakit.",
        },
    ]


# ============================================================
# Section-Based XAI
# ============================================================

def generate_section_importance(
    eeg_features: list,
    target_class: int,
    graph_sections: list,
):
    """
    Hitung kontribusi tiap section P1-P4 dengan occlusion.

    Cara kerja:
    - Ambil confidence awal.
    - Untuk setiap section, fitur pada section itu ditutup / dibuat 0.
    - Prediksi ulang.
    - Jika confidence turun besar, berarti section itu penting.
    """

    base_input = preprocess_eeg_array(eeg_features)
    base_result = run_eeg_inference(base_input)
    base_confidence = float(base_result["probabilities"][target_class])

    raw_features = np.array(eeg_features, dtype=np.float32)

    section_results = []

    for section in graph_sections:
        model_indices = section.get("model_indices", [])

        occluded_features = raw_features.copy()

        if model_indices:
            # Occlusion: area fitur section dibuat 0
            occluded_features[model_indices] = 0.0

        occluded_input = preprocess_eeg_array(occluded_features.tolist())
        occluded_result = run_eeg_inference(occluded_input)
        occluded_confidence = float(occluded_result["probabilities"][target_class])

        raw_importance = base_confidence - occluded_confidence
        safe_importance = max(float(raw_importance), 0.0)

        section_results.append({
            **section,
            "importance": round(safe_importance, 6),
            "raw_importance": round(float(raw_importance), 6),
            "base_confidence": round(base_confidence, 6),
            "occluded_confidence": round(occluded_confidence, 6),
        })

    max_importance = max(
        [item["importance"] for item in section_results],
        default=0.0,
    )

    # Normalisasi supaya 4 section punya warna relatif:
    # section paling penting = 100%
    normalized_results = []

    for item in section_results:
        if max_importance > 0:
            importance_percent = item["importance"] / max_importance * 100
        else:
            importance_percent = 0.0

        color_info = get_color_level_from_percent(importance_percent)

        normalized_results.append({
            **item,
            "importance_percent": round(float(importance_percent), 2),
            "level": color_info["level"],
            "color": color_info["color"],
            "label": color_info["label"],
            "description": color_info["description"],
        })

    normalized_results.sort(
        key=lambda item: item["importance_percent"],
        reverse=True,
    )

    return normalized_results, base_result


def build_section_summary(section_results: list):
    if not section_results:
        return {
            "primary_section": None,
            "summary": "Tidak ada section EEG yang dapat dianalisis.",
        }

    primary = section_results[0]

    return {
        "primary_section": {
            "name": primary["name"],
            "title": primary["title"],
            "level": primary["level"],
            "label": primary["label"],
            "color": primary["color"],
            "importance_percent": primary["importance_percent"],
            "start_sample": primary["start_sample"],
            "end_sample": primary["end_sample"],
        },
        "summary": (
            f"Section {primary['name']} memiliki kontribusi paling kuat "
            f"terhadap prediksi model, dengan nilai kontribusi relatif "
            f"{primary['importance_percent']}%."
        ),
    }


# ============================================================
# Main Service
# ============================================================

async def predict_eeg_with_xai(
    eeg_data: list,
    graph_sections: list | None = None,
):
    input_data = preprocess_eeg_array(eeg_data)

    prediction_result = run_eeg_inference(input_data)
    target_class = prediction_result["prediction_index"]

    if graph_sections:
        section_results, _ = generate_section_importance(
            eeg_features=eeg_data,
            target_class=target_class,
            graph_sections=graph_sections,
        )

        section_summary = build_section_summary(section_results)

        explanation_text = (
            f"Model memprediksi EEG sebagai {prediction_result['prediction_label']} "
            f"dengan confidence {prediction_result['confidence'] * 100:.2f}%. "
            f"{section_summary['summary']} Warna biru menunjukkan normal, hijau "
            f"observasi normal, kuning early warning, dan merah prediksi kuat penyakit."
        )

        return {
            "prediction_index": prediction_result["prediction_index"],
            "prediction_label": prediction_result["prediction_label"],
            "confidence": prediction_result["confidence"],
            "probabilities": prediction_result["probabilities"],
            "xai_method": "section_based_occlusion_sensitivity",
            "xai_method_label": "Section-Based Occlusion Sensitivity",
            "important_segments": section_results,
            "section_summary": section_summary,
            "color_legend": get_color_legend(),
            "explanation_text": explanation_text,
            "message": "EEG inference + section-based XAI berhasil",
        }

    # fallback kalau dipanggil tanpa graph_sections
    explanation_text = generate_eeg_xai_explanation(
        prediction_label=prediction_result["prediction_label"],
        confidence=prediction_result["confidence"],
        important_segments=[],
    )

    return {
        "prediction_index": prediction_result["prediction_index"],
        "prediction_label": prediction_result["prediction_label"],
        "confidence": prediction_result["confidence"],
        "probabilities": prediction_result["probabilities"],
        "xai_method": "eeg_inference_only",
        "xai_method_label": "EEG Inference Only",
        "important_segments": [],
        "section_summary": None,
        "color_legend": get_color_legend(),
        "explanation_text": explanation_text,
        "message": "EEG inference berhasil",
    }