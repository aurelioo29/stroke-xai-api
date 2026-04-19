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
        "prediction_label": CLASS_LABELS.get(prediction_index, f"class_{prediction_index}"),
        "confidence": confidence,
        "probabilities": probs.tolist(),
    }


def generate_eeg_occlusion_segments(
    eeg_array: list,
    target_class: int,
    segment_size: int = 150,
    stride: int = 75,
):
    base_input = preprocess_eeg_array(eeg_array)
    base_result = run_eeg_inference(base_input)
    base_confidence = base_result["probabilities"][target_class]

    signal = np.array(eeg_array, dtype=np.float32)
    important_segments = []

    for start in range(0, len(signal) - segment_size + 1, stride):
        end = start + segment_size

        occluded_signal = signal.copy()
        occluded_signal[start:end] = 0.0

        occluded_input = preprocess_eeg_array(occluded_signal.tolist())
        result = run_eeg_inference(occluded_input)
        occluded_confidence = result["probabilities"][target_class]

        importance = float(base_confidence - occluded_confidence)

        important_segments.append({
            "start": start,
            "end": end,
            "importance": round(importance, 6),
        })

    important_segments.sort(key=lambda x: x["importance"], reverse=True)
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
        "important_segments": top_segments,
        "explanation_text": explanation_text,
        "message": "EEG inference + XAI berhasil"
    }