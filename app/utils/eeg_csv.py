import csv
import io
import re
from fastapi import UploadFile, HTTPException


EXCLUDE_COLUMNS = ["label", "subject", "trial"]


def is_number(value):
    try:
        float(value)
        return True
    except Exception:
        return False


def extract_channel_graph(row: dict, channel: int = 1):
    pattern = re.compile(rf"^ch{channel}_s(\d+)$")

    points = []

    for key, value in row.items():
        match = pattern.match(key)

        if not match:
            continue

        if not is_number(value):
            continue

        sample = int(match.group(1))

        points.append({
            "sample": sample,
            "value": float(value),
            "channel": f"ch{channel}",
        })

    points.sort(key=lambda item: item["sample"])
    return points


async def read_eeg_csv(
    file: UploadFile,
    row_index: int = 0,
    graph_channel: int = 1,
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="File EEG wajib berformat .csv",
        )

    content = await file.read()

    try:
        text = content.decode("utf-8-sig")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="CSV tidak bisa dibaca",
        )

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)

    if not reader.fieldnames:
        raise HTTPException(
            status_code=400,
            detail="CSV kosong atau header tidak valid",
        )

    if not rows:
        raise HTTPException(
            status_code=400,
            detail="CSV tidak memiliki data",
        )

    if row_index < 0 or row_index >= len(rows):
        raise HTTPException(
            status_code=400,
            detail=f"Row index {row_index} tidak ditemukan. Total row: {len(rows)}",
        )

    row = rows[row_index]

    # Ambil fitur EEG asli, exclude metadata
    feature_columns = [
        col for col in reader.fieldnames
        if col not in EXCLUDE_COLUMNS and is_number(row.get(col))
    ]

    features = [float(row[col]) for col in feature_columns]

    # FIX:
    # Model kamu dilatih dengan 16385 fitur karena saat training
    # kolom "trial" tidak ikut di-drop.
    # Jadi kalau CSV hanya punya 16384 fitur EEG asli, tambahkan trial.
    if len(features) == 16384:
        trial_value = row.get("trial")

        if trial_value is not None and is_number(trial_value):
            features.append(float(trial_value))
        else:
            features.append(float(row_index + 1))

    if len(features) != 16385:
        raise HTTPException(
            status_code=400,
            detail=f"Jumlah fitur EEG harus 16385, tetapi ditemukan {len(features)}",
        )

    graph_data = extract_channel_graph(row, channel=graph_channel)

    if not graph_data:
        raise HTTPException(
            status_code=400,
            detail=f"Data grafik untuk channel ch{graph_channel} tidak ditemukan",
        )

    return {
        "model_input": features,
        "graph_data": graph_data,
        "feature_count": len(features),
        "selected_row": row_index,
        "selected_channel": f"ch{graph_channel}",
        "uploaded_filename": file.filename,
    }