import os
import requests

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()


def build_mri_prompt(prediction_label: str, confidence: float, xai_method: str):
    confidence_percent = round(confidence * 100, 2)

    return f"""
Anda adalah asisten penjelasan klinis untuk alat bantu diagnosis neurologi.

Tugas Anda:
- Jelaskan hasil model dalam bahasa Indonesia yang formal, jelas, dan mudah dipahami dokter umum.
- Jangan membuat diagnosis baru.
- Jangan memberi rekomendasi obat atau terapi.
- Jelaskan bahwa sistem ini adalah alat bantu klinis, bukan pengganti dokter.

Data hasil:
- Prediksi MRI: {prediction_label}
- Tingkat keyakinan: {confidence_percent}%
- Metode explainability: {xai_method}
- Arti heatmap:
  - Biru: kontribusi rendah
  - Hijau: kontribusi ringan hingga sedang
  - Kuning: kontribusi sedang hingga tinggi
  - Merah: kontribusi paling tinggi terhadap hasil prediksi

Buat 1 paragraf penjelasan klinis yang natural, tidak kaku, dan tidak terasa seperti template.
""".strip()


def build_multimodal_prompt(
    mri_label: str,
    eeg_label: str,
    final_label: str,
    confidence: float,
    xai_method: str,
):
    confidence_percent = round(confidence * 100, 2)

    return f"""
Anda adalah asisten penjelasan klinis untuk alat bantu diagnosis neurologi berbasis AI.

Tugas Anda:
- Jelaskan hasil analisis multimodal dalam bahasa Indonesia yang formal, jelas, dan mudah dipahami dokter umum.
- Jangan membuat diagnosis baru.
- Jangan memberi rekomendasi obat atau terapi.
- Tegaskan bahwa sistem ini adalah alat bantu klinis, bukan pengganti penilaian dokter.

Data hasil:
- Hasil MRI: {mri_label}
- Hasil EEG: {eeg_label}
- Prediksi akhir multimodal: {final_label}
- Tingkat keyakinan akhir: {confidence_percent}%
- Metode XAI MRI: {xai_method}
- Metode fusion: late fusion dengan pembobotan MRI 60% dan EEG 40%

Buat 1 paragraf penjelasan klinis yang natural, informatif, dan tidak terasa seperti template.
""".strip()


def _generate_openrouter(prompt: str):
    api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY tidak ditemukan di .env")

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "stroke-xai-api",
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "Anda membantu menjelaskan output AI klinis."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.4,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_llm_mri_explanation(
    prediction_label: str,
    confidence: float,
    xai_method: str,
):
    prompt = build_mri_prompt(
        prediction_label=prediction_label,
        confidence=confidence,
        xai_method=xai_method,
    )
    return _generate_openrouter(prompt)


def generate_llm_multimodal_explanation(
    mri_label: str,
    eeg_label: str,
    final_label: str,
    confidence: float,
    xai_method: str,
):
    prompt = build_multimodal_prompt(
        mri_label=mri_label,
        eeg_label=eeg_label,
        final_label=final_label,
        confidence=confidence,
        xai_method=xai_method,
    )
    return _generate_openrouter(prompt)