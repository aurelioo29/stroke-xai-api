from fastapi import UploadFile
from app.core.config import mri_session
from app.core.class_labels import CLASS_LABELS
from app.utils.preprocess_mri import preprocess_mri_image


async def predict_mri(file: UploadFile):
    contents = await file.read()

    input_data = preprocess_mri_image(contents)

    input_name = mri_session.get_inputs()[0].name
    output_name = mri_session.get_outputs()[0].name

    outputs = mri_session.run([output_name], {input_name: input_data})
    probs = outputs[0][0]

    prediction_index = int(probs.argmax())
    prediction_label = CLASS_LABELS.get(prediction_index, f"class_{prediction_index}")
    confidence = float(probs.max())

    return {
        "prediction_index": prediction_index,
        "prediction_label": prediction_label,
        "confidence": confidence,
        "probabilities": probs.tolist(),
        "message": "MRI inference berhasil"
    }