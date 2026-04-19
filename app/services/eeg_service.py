from app.core.config import eeg_session
from app.core.class_labels import CLASS_LABELS
from app.utils.preprocess_eeg import preprocess_eeg_array
from app.utils.explanation import generate_eeg_explanation


async def predict_eeg(eeg_data: list):
    input_data = preprocess_eeg_array(eeg_data)

    input_name = eeg_session.get_inputs()[0].name
    output_name = eeg_session.get_outputs()[0].name

    outputs = eeg_session.run([output_name], {input_name: input_data})
    probs = outputs[0][0]

    prediction_index = int(probs.argmax())
    prediction_label = CLASS_LABELS.get(prediction_index, f"class_{prediction_index}")
    confidence = float(probs.max())

    explanation_text = generate_eeg_explanation(
        prediction_label=prediction_label,
        confidence=confidence,
    )

    return {
        "prediction_index": prediction_index,
        "prediction_label": prediction_label,
        "confidence": confidence,
        "probabilities": probs.tolist(),
        "explanation_text": explanation_text,
        "message": "EEG inference berhasil"
    }