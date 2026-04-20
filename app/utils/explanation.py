def generate_mri_explanation(prediction_label: str, confidence: float):
    confidence_percent = round(confidence * 100, 2)

    return (
        f"Sistem memprediksi kemungkinan {prediction_label} dengan tingkat keyakinan "
        f"{confidence_percent}%. Visualisasi explainability dibangun menggunakan metode "
        f"occlusion sensitivity untuk menunjukkan area citra yang paling memengaruhi "
        f"keputusan model. Pada heatmap, warna biru menunjukkan kontribusi rendah, "
        f"warna hijau dan kuning menunjukkan kontribusi sedang, sedangkan warna merah "
        f"menunjukkan area dengan kontribusi paling tinggi terhadap hasil prediksi. "
        f"Area dengan warna yang lebih hangat dapat digunakan sebagai penanda visual "
        f"untuk membantu dokter memfokuskan evaluasi pada bagian citra yang paling relevan. "
        f"Hasil ini bersifat sebagai alat bantu klinis dan tidak menggantikan interpretasi "
        f"dokter, pembacaan radiologi, maupun korelasi dengan kondisi klinis pasien."
    )


def generate_mri_heatmap_legend():
    return {
        "blue": "Kontribusi rendah terhadap keputusan model",
        "green": "Kontribusi ringan hingga sedang",
        "yellow": "Kontribusi sedang hingga tinggi",
        "red": "Kontribusi paling tinggi terhadap hasil prediksi",
    }


def generate_mri_clinical_note(prediction_label: str):
    return (
        f"Prediksi {prediction_label} perlu dipahami sebagai dukungan analisis visual. "
        f"Dokter tetap perlu mengonfirmasi hasil melalui evaluasi klinis, pembacaan radiologi, "
        f"riwayat pasien, serta pemeriksaan penunjang lain bila diperlukan."
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
        f"Hasil akhir analisis multimodal menunjukkan kemungkinan {final_label} "
        f"dengan tingkat keyakinan {confidence_percent}%. Model MRI memberikan hasil "
        f"{mri_label}, sedangkan model EEG memberikan hasil {eeg_label}. "
        f"Keputusan akhir diperoleh melalui metode late fusion dengan pembobotan MRI "
        f"sebesar 60% dan EEG sebesar 40%, sehingga hasil akhir lebih dipengaruhi oleh "
        f"analisis citra MRI. Hasil ini dirancang sebagai alat bantu klinis untuk "
        f"memperkuat evaluasi dokter dan bukan sebagai pengganti penilaian klinis langsung."
    )