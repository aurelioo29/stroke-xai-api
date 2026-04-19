def generate_mri_explanation(prediction_label: str, confidence: float):
    confidence_percent = round(confidence * 100, 2)

    return (
        f"Model memprediksi {prediction_label} dengan tingkat keyakinan "
        f"{confidence_percent}%. Berdasarkan metode occlusion sensitivity, "
        f"area dengan intensitas warna tinggi pada heatmap menunjukkan bagian "
        f"citra yang paling berkontribusi terhadap keputusan model. "
        f"Semakin merah area tersebut, semakin besar pengaruhnya terhadap hasil klasifikasi."
    )


def generate_eeg_explanation(prediction_label: str, confidence: float):
    confidence_percent = round(confidence * 100, 2)

    return (
        f"Model EEG memprediksi {prediction_label} dengan tingkat keyakinan "
        f"{confidence_percent}%. Prediksi ini dihasilkan dari analisis pola sinyal EEG "
        f"sepanjang 1500 titik data. Beberapa bagian sinyal memberikan kontribusi lebih besar "
        f"terhadap keputusan model, meskipun belum divisualisasikan secara spesifik."
    )


def generate_eeg_xai_explanation(
    prediction_label: str,
    confidence: float,
    important_segments: list,
):
    confidence_percent = round(confidence * 100, 2)

    if not important_segments:
        return (
            f"Model EEG memprediksi {prediction_label} dengan tingkat keyakinan "
            f"{confidence_percent}%. Analisis explainability belum menemukan segmen "
            f"sinyal yang dominan secara signifikan."
        )

    top_segments_text = ", ".join(
        [f"{seg['start']}-{seg['end']}" for seg in important_segments[:3]]
    )

    return (
        f"Model EEG memprediksi {prediction_label} dengan tingkat keyakinan "
        f"{confidence_percent}%. Berdasarkan metode occlusion segment sensitivity, "
        f"segmen sinyal pada rentang {top_segments_text} memberikan kontribusi "
        f"terbesar terhadap keputusan model."
    )


def generate_multimodal_explanation(
    mri_label: str,
    eeg_label: str,
    final_label: str,
    confidence: float,
):
    confidence_percent = round(confidence * 100, 2)

    return (
        f"Hasil akhir prediksi adalah {final_label} dengan tingkat keyakinan "
        f"{confidence_percent}%. Model MRI memprediksi {mri_label}, sedangkan model EEG "
        f"memprediksi {eeg_label}. Hasil akhir diperoleh menggunakan metode "
        f"late fusion dengan pembobotan MRI sebesar 60% dan EEG sebesar 40%, "
        f"sehingga keputusan akhir lebih dipengaruhi oleh hasil analisis citra MRI."
    )