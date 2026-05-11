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


# ============================================================
# Output Directories
# ============================================================

HEATMAP_DIR = Path("outputs/heatmaps")
OVERLAY_DIR = Path("outputs/overlays")
FOCUS_DIR = Path("outputs/focus")

HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
FOCUS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Label Config
# ============================================================

DISEASE_LABELS = {"hemorrhagic", "ischemic"}


# ============================================================
# Inference Helper
# ============================================================

def run_mri_inference(input_data: np.ndarray):
    input_name = mri_session.get_inputs()[0].name
    output_name = mri_session.get_outputs()[0].name

    outputs = mri_session.run([output_name], {input_name: input_data})
    probs = outputs[0][0]
    probs = np.asarray(probs, dtype=np.float32)

    # Safety:
    # Jika output model masih logits, ubah ke probability.
    # Kalau output sudah probability, bagian ini tidak akan mengganggu.
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
# Occlusion Sensitivity
# ============================================================

def generate_occlusion_heatmap(
    pil_image: Image.Image,
    target_class: int,
    patch_size: int = 32,
    stride: int = 16,
):
    """
    Occlusion sensitivity.

    Cara kerja:
    - Area gambar ditutup sebagian.
    - Model melakukan prediksi ulang.
    - Jika confidence target class turun, berarti area itu penting
      untuk keputusan model.
    """

    base_input = preprocess_mri_pil_image(pil_image)
    base_result = run_mri_inference(base_input)
    base_confidence = float(base_result["probabilities"][target_class])

    image_np = np.array(pil_image).astype(np.float32) / 255.0
    h, w, _ = image_np.shape

    heatmap = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    # Pakai mean pixel, bukan hitam total.
    # Mask hitam sering bikin artefak terlalu agresif.
    mean_pixel = image_np.mean(axis=(0, 1), keepdims=True)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            occluded = image_np.copy()
            occluded[y:y + patch_size, x:x + patch_size] = mean_pixel

            occluded_input = np.expand_dims(occluded, axis=0)
            result = run_mri_inference(occluded_input)

            occluded_confidence = float(result["probabilities"][target_class])
            importance = base_confidence - occluded_confidence

            heatmap[y:y + patch_size, x:x + patch_size] += importance
            counts[y:y + patch_size, x:x + patch_size] += 1

    counts[counts == 0] = 1
    heatmap = heatmap / counts

    # Hanya kontribusi positif yang dianggap penting.
    heatmap = np.maximum(heatmap, 0)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


# ============================================================
# Clinical Colormaps
# ============================================================

def get_normal_blue_colormap():
    """
    Untuk prediksi NORMAL:
    - Biru = normal
    - Biru semakin pekat = semakin kuat mendukung prediksi normal

    Penting:
    Alpha di dalam colormap dibuat 1.0 agar biru tidak hilang.
    Transparansi overlay diatur lewat plt.imshow(alpha=...).
    """

    return LinearSegmentedColormap.from_list(
        "normal_blue_scale_visible",
        [
            (0.0, (0.65, 0.82, 1.00, 1.0)),   # biru muda
            (0.35, (0.25, 0.55, 1.00, 1.0)),  # biru sedang
            (0.70, (0.00, 0.25, 1.00, 1.0)),  # biru kuat
            (1.0, (0.00, 0.05, 0.85, 1.0)),   # biru pekat
        ],
    )


def get_disease_risk_colormap():
    """
    Untuk prediksi PENYAKIT:
    - Biru   = normal / kontribusi sangat rendah
    - Hijau  = observasi normal / kontribusi rendah
    - Kuning = early warning menuju penyakit
    - Merah  = prediksi kuat penyakit

    Penting:
    Semua warna dibuat terlihat.
    Transparansi overlay diatur lewat plt.imshow(alpha=...).
    """

    return LinearSegmentedColormap.from_list(
        "clinical_disease_risk_scale_visible",
        [
            (0.0, (0.20, 0.45, 1.00, 1.0)),   # biru terlihat
            (0.30, (0.00, 0.70, 1.00, 1.0)),  # biru-cyan
            (0.50, (0.00, 0.85, 0.25, 1.0)),  # hijau
            (0.72, (1.00, 0.90, 0.00, 1.0)),  # kuning
            (1.0, (1.00, 0.00, 0.00, 1.0)),   # merah
        ],
    )


def get_raw_academic_colormap():
    """
    Raw academic heatmap.
    Tetap mengikuti standar:
    biru -> hijau -> kuning -> merah.
    """

    return LinearSegmentedColormap.from_list(
        "academic_blue_green_yellow_red_visible",
        [
            (0.0, "#3B73FF"),
            (0.35, "#00D46A"),
            (0.70, "#FFE600"),
            (1.0, "#FF0000"),
        ],
    )


# ============================================================
# Zone Analysis
# ============================================================

def build_clinical_zone_analysis(
    heatmap: np.ndarray,
    prediction_label: str,
):
    is_disease = prediction_label in DISEASE_LABELS

    if prediction_label == "normal":
        return {
            "display_heatmap": heatmap,
            "is_disease_prediction": False,
            "zone_label": "normal_zone",
            "active_area_percent": 0.0,
            "color_scale": [
                {
                    "color": "blue",
                    "meaning": (
                        "Normal. Biru semakin pekat berarti semakin kuat "
                        "mendukung prediksi normal."
                    ),
                },
            ],
            "interpretation": (
                "Prediksi normal. Area biru yang semakin pekat menunjukkan "
                "bagian citra yang semakin kuat mendukung prediksi normal."
            ),
        }

    positive_values = heatmap[heatmap > 0]

    if positive_values.size == 0:
        return {
            "display_heatmap": np.zeros_like(heatmap, dtype=np.float32),
            "is_disease_prediction": is_disease,
            "zone_label": "no_clear_focus",
            "active_area_percent": 0.0,
            "color_scale": [
                {
                    "color": "blue",
                    "meaning": "Normal / kontribusi sangat rendah",
                },
                {
                    "color": "green",
                    "meaning": "Observasi normal / kontribusi rendah",
                },
                {
                    "color": "yellow",
                    "meaning": "Early warning menuju prediksi penyakit",
                },
                {
                    "color": "red",
                    "meaning": "Prediksi kuat penyakit",
                },
            ],
            "interpretation": (
                "Prediksi penyakit, tetapi tidak ditemukan zona fokus yang kuat "
                "pada heatmap."
            ),
        }

    # Area aktif dihitung dari zona yang mulai masuk warning.
    # 70 percentile = area kuning/merah mulai dianggap warning/risk area.
    threshold = float(np.percentile(positive_values, 70))
    active_mask = heatmap >= threshold
    active_area_percent = float(active_mask.sum() / heatmap.size * 100)

    return {
        "display_heatmap": heatmap,
        "is_disease_prediction": True,
        "zone_label": f"{prediction_label}_risk_zone",
        "active_area_percent": round(active_area_percent, 4),
        "color_scale": [
            {
                "color": "blue",
                "meaning": "Normal / kontribusi sangat rendah",
            },
            {
                "color": "green",
                "meaning": "Observasi normal / kontribusi rendah",
            },
            {
                "color": "yellow",
                "meaning": "Early warning / mulai mengarah penyakit",
            },
            {
                "color": "red",
                "meaning": "Prediksi kuat penyakit",
            },
        ],
        "interpretation": (
            f"Prediksi {prediction_label}. Biru menunjukkan area normal atau kontribusi "
            "sangat rendah. Hijau menunjukkan area observasi normal. Kuning menunjukkan "
            "area early warning. Merah menunjukkan area yang paling kuat mendukung "
            "prediksi penyakit."
        ),
    }


# ============================================================
# Save Images
# ============================================================

def save_mri_xai_images(
    pil_image: Image.Image,
    heatmap: np.ndarray,
    prediction_label: str,
):
    file_id = uuid.uuid4().hex

    raw_heatmap_path = HEATMAP_DIR / f"raw_heatmap_{file_id}.png"
    clinical_overlay_path = OVERLAY_DIR / f"clinical_overlay_{file_id}.png"
    disease_focus_path = FOCUS_DIR / f"disease_focus_{file_id}.png"

    zone_result = build_clinical_zone_analysis(
        heatmap=heatmap,
        prediction_label=prediction_label,
    )

    display_heatmap = zone_result["display_heatmap"]

    # ------------------------------------------------------------
    # 1. Raw academic heatmap
    # ------------------------------------------------------------
    # Peta teknis tetap memperlihatkan MRI + warna heatmap.
    # Biru tidak hilang karena alpha colormap = 1.0.
    # Transparansi global diatur lewat alpha=0.42.
    plt.figure(figsize=(6, 6))
    plt.imshow(pil_image)

    if prediction_label == "normal":
        plt.imshow(
            heatmap,
            cmap=get_normal_blue_colormap(),
            vmin=0,
            vmax=1,
            alpha=0.42,
        )
    else:
        plt.imshow(
            heatmap,
            cmap=get_disease_risk_colormap(),
            vmin=0,
            vmax=1,
            alpha=0.42,
        )

    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(raw_heatmap_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    # ------------------------------------------------------------
    # 2. Clinical overlay utama
    # ------------------------------------------------------------
    # Ini gambar utama yang tampil di frontend.
    # Alpha 0.38 menjaga warna tetap terlihat tanpa menutup struktur otak.
    plt.figure(figsize=(6, 6))
    plt.imshow(pil_image)

    if prediction_label == "normal":
        plt.imshow(
            display_heatmap,
            cmap=get_normal_blue_colormap(),
            vmin=0,
            vmax=1,
            alpha=0.38,
        )
    else:
        plt.imshow(
            display_heatmap,
            cmap=get_disease_risk_colormap(),
            vmin=0,
            vmax=1,
            alpha=0.38,
        )

    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(clinical_overlay_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    # ------------------------------------------------------------
    # 3. Disease focus / clinical focus image
    # ------------------------------------------------------------
    # Disimpan terpisah kalau nanti ingin dipakai di frontend/history.
    plt.figure(figsize=(6, 6))
    plt.imshow(pil_image)

    if prediction_label == "normal":
        plt.imshow(
            display_heatmap,
            cmap=get_normal_blue_colormap(),
            vmin=0,
            vmax=1,
            alpha=0.38,
        )
    else:
        plt.imshow(
            display_heatmap,
            cmap=get_disease_risk_colormap(),
            vmin=0,
            vmax=1,
            alpha=0.38,
        )

    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(disease_focus_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return {
        "raw_heatmap_url": f"/outputs/heatmaps/{raw_heatmap_path.name}",
        "heatmap_url": f"/outputs/heatmaps/{raw_heatmap_path.name}",
        "overlay_url": f"/outputs/overlays/{clinical_overlay_path.name}",
        "disease_focus_url": f"/outputs/focus/{disease_focus_path.name}",
        "zone_analysis": {
            "is_disease_prediction": zone_result["is_disease_prediction"],
            "zone_label": zone_result["zone_label"],
            "active_area_percent": zone_result["active_area_percent"],
            "color_scale": zone_result["color_scale"],
            "interpretation": zone_result["interpretation"],
        },
    }


# ============================================================
# Main Service
# ============================================================

async def predict_mri_with_xai(file):
    contents = await file.read()

    pil_image = load_and_resize_mri_image(contents)
    input_data = preprocess_mri_pil_image(pil_image)

    pred = run_mri_inference(input_data)

    target_class = pred["prediction_index"]
    prediction_label = pred["prediction_label"]

    heatmap = generate_occlusion_heatmap(
        pil_image=pil_image,
        target_class=target_class,
        patch_size=32,
        stride=16,
    )

    images = save_mri_xai_images(
        pil_image=pil_image,
        heatmap=heatmap,
        prediction_label=prediction_label,
    )

    explanation_text = generate_mri_explanation(
        prediction_label=prediction_label,
        confidence=pred["confidence"],
    )
    explanation_source = "rule_based"

    if generate_llm_mri_explanation is not None:
        try:
            explanation_text = generate_llm_mri_explanation(
                prediction_label=prediction_label,
                confidence=pred["confidence"],
                xai_method="clinical_color_zone_occlusion_sensitivity",
            )
            explanation_source = "llm"
        except Exception as e:
            print("LLM MRI explanation error:", e)

    if prediction_label == "normal":
        clinical_message = (
            "MRI diprediksi normal. Area biru yang semakin pekat menunjukkan "
            "bagian citra yang semakin kuat mendukung prediksi normal."
        )
    else:
        clinical_message = (
            f"MRI diprediksi {prediction_label}. Biru menunjukkan area normal, "
            "hijau menunjukkan observasi normal, kuning menunjukkan early warning, "
            "dan merah menunjukkan area yang paling kuat mendukung prediksi penyakit."
        )

    return {
        **pred,

        "raw_heatmap_url": images["raw_heatmap_url"],
        "heatmap_url": images["heatmap_url"],
        "overlay_url": images["overlay_url"],
        "disease_focus_url": images["disease_focus_url"],

        "zone_analysis": images["zone_analysis"],

        "xai_method": "clinical_color_zone_occlusion_sensitivity",
        "heatmap_legend": generate_mri_heatmap_legend(),
        "clinical_note": generate_mri_clinical_note(prediction_label),
        "clinical_message": clinical_message,

        "explanation_text": explanation_text,
        "explanation_source": explanation_source,

        "message": "MRI inference + clinical color zone XAI berhasil",
    }