import uuid
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from app.core.config import mri_session
from app.core.class_labels import CLASS_LABELS
from app.utils.preprocess_mri import (
    load_and_resize_mri_image,
    preprocess_mri_pil_image,
)
from app.utils.explanation import (
    generate_mri_explanation,
    generate_mri_heatmap_legend,
    generate_mri_clinical_note,
)

try:
    from app.services.llm_explanation_service import generate_llm_mri_explanation
except Exception:
    generate_llm_mri_explanation = None


HEATMAP_DIR = Path("outputs/heatmaps")
OVERLAY_DIR = Path("outputs/overlays")
FOCUS_DIR = Path("outputs/focus")

HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
FOCUS_DIR.mkdir(parents=True, exist_ok=True)


DISEASE_LABELS = {"hemorrhagic", "ischemic"}


def run_mri_inference(input_data: np.ndarray):
    input_name = mri_session.get_inputs()[0].name
    output_name = mri_session.get_outputs()[0].name

    outputs = mri_session.run([output_name], {input_name: input_data})
    probs = outputs[0][0]

    prediction_index = int(np.argmax(probs))
    confidence = float(np.max(probs))

    return {
        "prediction_index": prediction_index,
        "prediction_label": CLASS_LABELS.get(
            prediction_index,
            f"class_{prediction_index}"
        ),
        "confidence": confidence,
        "probabilities": probs.tolist(),
    }


def generate_occlusion_heatmap(
    pil_image,
    target_class,
    patch_size=32,
    stride=16,
):
    base_input = preprocess_mri_pil_image(pil_image)
    base_result = run_mri_inference(base_input)
    base_confidence = float(base_result["probabilities"][target_class])

    image_np = np.array(pil_image).astype(np.float32) / 255.0
    h, w, _ = image_np.shape

    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            occluded = image_np.copy()

            # masking area image
            occluded[y:y + patch_size, x:x + patch_size] = 0.0

            occluded_input = np.expand_dims(occluded, axis=0)
            result = run_mri_inference(occluded_input)

            occluded_conf = float(result["probabilities"][target_class])
            importance = base_confidence - occluded_conf

            heatmap[y:y + patch_size, x:x + patch_size] += importance
            counts[y:y + patch_size, x:x + patch_size] += 1

    counts[counts == 0] = 1
    heatmap = heatmap / counts

    # hanya kontribusi positif yang dianggap mendukung prediksi
    heatmap = np.maximum(heatmap, 0)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def build_disease_focus_mask(
    heatmap: np.ndarray,
    prediction_label: str,
    percentile: int = 85,
):
    """
    Untuk revisi DPL:
    - Merah hanya fokus pada area yang paling mendukung penyakit prediksi.
    - Area normal / rendah kontribusi tidak perlu ditonjolkan.
    """

    if prediction_label not in DISEASE_LABELS:
        return {
            "mask": np.zeros_like(heatmap, dtype=np.float32),
            "threshold": None,
            "active_area_percent": 0.0,
            "is_disease_prediction": False,
        }

    positive_values = heatmap[heatmap > 0]

    if positive_values.size == 0:
        return {
            "mask": np.zeros_like(heatmap, dtype=np.float32),
            "threshold": None,
            "active_area_percent": 0.0,
            "is_disease_prediction": True,
        }

    threshold = float(np.percentile(positive_values, percentile))

    mask = np.where(heatmap >= threshold, heatmap, 0.0).astype(np.float32)

    if mask.max() > 0:
        mask = mask / mask.max()

    active_area_percent = float((mask > 0).sum() / mask.size * 100)

    return {
        "mask": mask,
        "threshold": threshold,
        "active_area_percent": round(active_area_percent, 4),
        "is_disease_prediction": True,
    }


def get_red_focus_colormap():
    """
    Colormap transparan -> merah.
    Area rendah tidak terlihat dominan.
    Area tinggi menjadi merah sebagai indikasi lokasi penyakit.
    """
    return LinearSegmentedColormap.from_list(
        "transparent_red_focus",
        [
            (0.0, (1.0, 0.0, 0.0, 0.0)),
            (0.3, (1.0, 0.0, 0.0, 0.20)),
            (0.7, (1.0, 0.0, 0.0, 0.55)),
            (1.0, (1.0, 0.0, 0.0, 0.85)),
        ],
    )


def save_heatmap_images(
    pil_image,
    heatmap,
    prediction_label,
):
    file_id = uuid.uuid4().hex

    heatmap_path = HEATMAP_DIR / f"heatmap_{file_id}.png"
    overlay_path = OVERLAY_DIR / f"overlay_{file_id}.png"
    focus_path = FOCUS_DIR / f"focus_{file_id}.png"

    focus_result = build_disease_focus_mask(
        heatmap=heatmap,
        prediction_label=prediction_label,
        percentile=85,
    )

    focus_mask = focus_result["mask"]

    # 1. Raw heatmap untuk debugging / akademik
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(heatmap_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    # 2. Overlay lama, masih boleh dipakai sebagai visual umum
    plt.figure(figsize=(6, 6))
    plt.imshow(pil_image)
    plt.imshow(heatmap, cmap="jet", alpha=0.35)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(overlay_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    # 3. Disease focus overlay: merah = area prediksi penyakit
    plt.figure(figsize=(6, 6))
    plt.imshow(pil_image)

    if focus_result["is_disease_prediction"] and focus_mask.max() > 0:
        plt.imshow(
            focus_mask,
            cmap=get_red_focus_colormap(),
            vmin=0,
            vmax=1,
            alpha=1.0,
        )

    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(focus_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return {
        "heatmap_url": f"/outputs/heatmaps/{heatmap_path.name}",
        "overlay_url": f"/outputs/overlays/{overlay_path.name}",
        "disease_focus_url": f"/outputs/focus/{focus_path.name}",
        "disease_focus_summary": {
            "is_disease_prediction": focus_result["is_disease_prediction"],
            "threshold_percentile": 85,
            "threshold_value": focus_result["threshold"],
            "active_area_percent": focus_result["active_area_percent"],
            "meaning": (
                "Area merah menunjukkan bagian citra yang paling berkontribusi "
                "terhadap prediksi penyakit."
                if focus_result["is_disease_prediction"]
                else "Prediksi normal, sehingga tidak ada area penyakit yang ditonjolkan."
            ),
        },
    }


async def predict_mri_with_xai(file):
    contents = await file.read()

    pil_image = load_and_resize_mri_image(contents)
    input_data = preprocess_mri_pil_image(pil_image)

    pred = run_mri_inference(input_data)
    target_class = pred["prediction_index"]

    heatmap = generate_occlusion_heatmap(
        pil_image=pil_image,
        target_class=target_class,
        patch_size=32,
        stride=16,
    )

    images = save_heatmap_images(
        pil_image=pil_image,
        heatmap=heatmap,
        prediction_label=pred["prediction_label"],
    )

    explanation_text = generate_mri_explanation(
        prediction_label=pred["prediction_label"],
        confidence=pred["confidence"],
    )
    explanation_source = "rule_based"

    if generate_llm_mri_explanation is not None:
        try:
            explanation_text = generate_llm_mri_explanation(
                prediction_label=pred["prediction_label"],
                confidence=pred["confidence"],
                xai_method="occlusion_sensitivity",
            )
            explanation_source = "llm"
        except Exception as e:
            print("LLM MRI explanation error:", e)

    return {
        **pred,
        "heatmap_url": images["heatmap_url"],
        "overlay_url": images["overlay_url"],
        "disease_focus_url": images["disease_focus_url"],
        "disease_focus_summary": images["disease_focus_summary"],
        "xai_method": "occlusion_sensitivity",
        "heatmap_legend": generate_mri_heatmap_legend(),
        "clinical_note": generate_mri_clinical_note(pred["prediction_label"]),
        "explanation_text": explanation_text,
        "explanation_source": explanation_source,
        "message": "MRI inference + XAI berhasil",
    }