import uuid
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from app.core.config import mri_session
from app.core.class_labels import CLASS_LABELS
from app.utils.preprocess_mri import load_and_resize_mri_image, preprocess_mri_pil_image
from app.utils.explanation import generate_mri_explanation

HEATMAP_DIR = Path("outputs/heatmaps")
OVERLAY_DIR = Path("outputs/overlays")

HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)


def run_mri_inference(input_data: np.ndarray):
    input_name = mri_session.get_inputs()[0].name
    output_name = mri_session.get_outputs()[0].name

    outputs = mri_session.run([output_name], {input_name: input_data})
    probs = outputs[0][0]

    prediction_index = int(np.argmax(probs))
    confidence = float(np.max(probs))

    return {
        "prediction_index": prediction_index,
        "prediction_label": CLASS_LABELS.get(prediction_index, f"class_{prediction_index}"),
        "confidence": confidence,
        "probabilities": probs.tolist(),
    }


def generate_occlusion_heatmap(
    pil_image: Image.Image,
    target_class: int,
    patch_size: int = 32,
    stride: int = 16,
):
    base_input = preprocess_mri_pil_image(pil_image)
    base_result = run_mri_inference(base_input)
    base_confidence = base_result["probabilities"][target_class]

    image_np = np.array(pil_image).astype(np.float32) / 255.0
    h, w, _ = image_np.shape

    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            occluded = image_np.copy()
            occluded[y:y + patch_size, x:x + patch_size, :] = 0.0

            occluded_input = np.expand_dims(occluded.astype(np.float32), axis=0)
            result = run_mri_inference(occluded_input)
            occluded_confidence = result["probabilities"][target_class]

            importance = base_confidence - occluded_confidence

            heatmap[y:y + patch_size, x:x + patch_size] += importance
            counts[y:y + patch_size, x:x + patch_size] += 1.0

    counts[counts == 0] = 1.0
    heatmap = heatmap / counts
    heatmap = np.maximum(heatmap, 0)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap, base_result


def save_heatmap_images(pil_image: Image.Image, heatmap: np.ndarray):
    file_id = uuid.uuid4().hex

    heatmap_path = HEATMAP_DIR / f"heatmap_{file_id}.png"
    overlay_path = OVERLAY_DIR / f"overlay_{file_id}.png"

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(heatmap_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(pil_image)
    plt.imshow(heatmap, cmap="jet", alpha=0.4)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(overlay_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return {
        "heatmap_url": f"/outputs/heatmaps/{heatmap_path.name}",
        "overlay_url": f"/outputs/overlays/{overlay_path.name}",
    }


async def predict_mri_with_xai(file):
    contents = await file.read()

    pil_image = load_and_resize_mri_image(contents, size=(224, 224))
    input_data = preprocess_mri_pil_image(pil_image)

    prediction_result = run_mri_inference(input_data)
    target_class = prediction_result["prediction_index"]

    heatmap, _ = generate_occlusion_heatmap(
        pil_image=pil_image,
        target_class=target_class,
        patch_size=32,
        stride=16,
    )

    image_paths = save_heatmap_images(pil_image, heatmap)

    explanation_text = generate_mri_explanation(
        prediction_label=prediction_result["prediction_label"],
        confidence=prediction_result["confidence"],
    )

    return {
        "prediction_index": prediction_result["prediction_index"],
        "prediction_label": prediction_result["prediction_label"],
        "confidence": prediction_result["confidence"],
        "probabilities": prediction_result["probabilities"],
        "heatmap_url": image_paths["heatmap_url"],
        "overlay_url": image_paths["overlay_url"],
        "xai_method": "occlusion_sensitivity",
        "explanation_text": explanation_text,
        "message": "MRI inference + XAI berhasil"
    }