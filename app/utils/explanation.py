def generate_mri_explanation(prediction_label: str, confidence: float):
    confidence_percent = round(confidence * 100, 2)

    if prediction_label == "normal":
        return (
            f"Model MRI memprediksi kondisi normal dengan tingkat keyakinan "
            f"{confidence_percent}%. Pada hasil XAI, sistem tidak menonjolkan area merah "
            f"karena tidak ada area citra yang diinterpretasikan sebagai lokasi penyakit "
            f"utama oleh model. Area dengan kontribusi rendah dapat diabaikan dan tetap "
            f"perlu dikonfirmasi melalui penilaian klinis serta pembacaan radiologi."
        )

    return (
        f"Model MRI memprediksi kemungkinan {prediction_label} dengan tingkat keyakinan "
        f"{confidence_percent}%. Berdasarkan metode occlusion sensitivity, area berwarna "
        f"merah menunjukkan bagian citra yang paling berkontribusi terhadap prediksi penyakit "
        f"tersebut. Area berwarna biru atau gelap menunjukkan kontribusi rendah dan dapat "
        f"diabaikan dalam interpretasi utama. Visualisasi ini bertujuan membantu dokter "
        f"memfokuskan perhatian pada area yang paling relevan, bukan menggantikan pembacaan "
        f"radiologi atau keputusan klinis."
    )


def generate_mri_heatmap_legend():
    return {
        "red": "Area paling berkontribusi terhadap prediksi penyakit",
        "yellow_or_green": "Area dengan kontribusi sedang terhadap keputusan model",
        "blue_or_dark": "Area kontribusi rendah / dapat diabaikan",
    }


def generate_mri_clinical_note(prediction_label: str):
    if prediction_label == "normal":
        return (
            "Prediksi normal menunjukkan bahwa model tidak menemukan area dominan yang "
            "mendukung kelas penyakit. Hasil tetap perlu dikonfirmasi oleh dokter dan "
            "pembacaan radiologi."
        )

    return (
        f"Area merah pada visualisasi XAI menunjukkan lokasi yang paling mendukung "
        f"prediksi {prediction_label}. Hasil ini bersifat alat bantu klinis dan harus "
        f"dikonfirmasi dengan evaluasi dokter, pembacaan radiologi, riwayat pasien, "
        f"dan pemeriksaan penunjang lain."
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
            f"{confidence_percent}%. Namun, analisis explainability belum menemukan "
            f"segmen sinyal yang menunjukkan kontribusi dominan secara signifikan."
        )

    top_segments_text = ", ".join(
        [f"{seg['start']}-{seg['end']}" for seg in important_segments[:3]]
    )

    max_drop = max(
        seg.get("confidence_drop_percent", 0.0)
        for seg in important_segments
    )

    if max_drop < 1.0:
        return (
            f"Model EEG memprediksi {prediction_label} dengan tingkat keyakinan "
            f"{confidence_percent}%. Berdasarkan metode occlusion segment sensitivity, "
            f"segmen seperti {top_segments_text} termasuk yang paling berpengaruh secara relatif, "
            f"namun penurunan confidence antar segmen masih sangat kecil. Hal ini menunjukkan "
            f"bahwa model belum menunjukkan fokus kuat pada satu rentang sinyal tertentu."
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