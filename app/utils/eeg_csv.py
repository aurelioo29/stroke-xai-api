import re
import pandas as pd
import numpy as np
from fastapi import UploadFile

from app.utils.preprocess_eeg import get_expected_feature_count


def natural_sample_sort(column_name: str):
    """
    Sort kolom seperti ch1_s1, ch1_s2, ..., ch1_s1000.
    Bukan sort string yang bikin ch1_s1000 muncul sebelum ch1_s2.
    """

    match = re.search(r"_s(\d+)$", str(column_name))

    if not match:
        return 0

    return int(match.group(1))


def get_channel_number(column_name: str):
    match = re.search(r"^ch(\d+)_s\d+$", str(column_name))

    if not match:
        return "?"

    return int(match.group(1))


def get_channel_sample_columns(columns, graph_channel: int):
    pattern = re.compile(rf"^ch{graph_channel}_s\d+$")

    channel_columns = [
        col for col in columns
        if pattern.match(str(col))
    ]

    channel_columns = sorted(
        channel_columns,
        key=natural_sample_sort,
    )

    return channel_columns


def select_model_feature_columns(df: pd.DataFrame):
    """
    Pilih kolom fitur yang jumlahnya cocok dengan scaler.

    Kenapa perlu begini?
    Karena model lama kamu kemungkinan dilatih dengan kolom trial ikut fitur,
    sehingga expected feature = 16385.

    Kalau trial dibuang, hasilnya jadi 16384 dan backend error.
    """

    expected_count = get_expected_feature_count()

    columns = list(df.columns)

    # Opsi 1:
    # Drop subject dan label saja.
    # trial tetap ikut.
    # Ini cocok kalau scaler lama dilatih dengan trial ikut fitur.
    drop_subject_label = [
        col for col in ["subject", "label"]
        if col in columns
    ]

    feature_columns_with_trial = [
        col for col in columns
        if col not in drop_subject_label
    ]

    if len(feature_columns_with_trial) == expected_count:
        return feature_columns_with_trial, {
            "mode": "with_trial_if_available",
            "dropped_columns": drop_subject_label,
            "expected_feature_count": expected_count,
            "actual_feature_count": len(feature_columns_with_trial),
        }

    # Opsi 2:
    # Drop subject, trial, label.
    # Ini cocok kalau scaler baru dilatih tanpa trial.
    drop_subject_trial_label = [
        col for col in ["subject", "trial", "label"]
        if col in columns
    ]

    feature_columns_without_trial = [
        col for col in columns
        if col not in drop_subject_trial_label
    ]

    if len(feature_columns_without_trial) == expected_count:
        return feature_columns_without_trial, {
            "mode": "without_trial",
            "dropped_columns": drop_subject_trial_label,
            "expected_feature_count": expected_count,
            "actual_feature_count": len(feature_columns_without_trial),
        }

    raise ValueError(
        f"Jumlah fitur CSV tidak cocok dengan scaler. "
        f"Scaler membutuhkan {expected_count} fitur. "
        f"Jika drop subject+label, fitur menjadi {len(feature_columns_with_trial)}. "
        f"Jika drop subject+trial+label, fitur menjadi {len(feature_columns_without_trial)}. "
        f"Cek kembali apakah CSV dan scaler berasal dari preprocessing training yang sama."
    )


def build_eeg_graph_sections(
    row: pd.Series,
    channel_columns: list,
    model_feature_columns: list,
    section_count: int = 4,
    section_size: int | None = None,
):
    total_samples = len(channel_columns)

    if total_samples == 0:
        return []

    sections = []

    # Mode fixed:
    # section_size=5:
    # P1 = s1-s5
    # P2 = s6-s10
    # P3 = s11-s15
    # P4 = s16-s20
    if section_size is not None and section_size > 0:
        for i in range(section_count):
            start_idx = i * section_size
            end_idx = min(start_idx + section_size, total_samples)

            if start_idx >= total_samples:
                break

            selected_columns = channel_columns[start_idx:end_idx]

            start_sample = natural_sample_sort(selected_columns[0])
            end_sample = natural_sample_sort(selected_columns[-1])

            model_indices = [
                model_feature_columns.index(col)
                for col in selected_columns
                if col in model_feature_columns
            ]

            data = [
                {
                    "sample": natural_sample_sort(col),
                    "value": float(row[col]),
                }
                for col in selected_columns
            ]

            channel_number = get_channel_number(selected_columns[0])

            sections.append({
                "name": f"P{i + 1}",
                "title": (
                    f"P{i + 1}: ch{channel_number}_s{start_sample} "
                    f"- ch{channel_number}_s{end_sample}"
                ),
                "start_sample": start_sample,
                "end_sample": end_sample,
                "columns": selected_columns,
                "model_indices": model_indices,
                "data": data,
            })

        return sections

    # Mode auto:
    # Semua sample dibagi rata menjadi P1-P4.
    chunk_size = int(np.ceil(total_samples / section_count))

    for i in range(section_count):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_samples)

        if start_idx >= total_samples:
            break

        selected_columns = channel_columns[start_idx:end_idx]

        start_sample = natural_sample_sort(selected_columns[0])
        end_sample = natural_sample_sort(selected_columns[-1])

        model_indices = [
            model_feature_columns.index(col)
            for col in selected_columns
            if col in model_feature_columns
        ]

        data = [
            {
                "sample": natural_sample_sort(col),
                "value": float(row[col]),
            }
            for col in selected_columns
        ]

        channel_number = get_channel_number(selected_columns[0])

        sections.append({
            "name": f"P{i + 1}",
            "title": (
                f"P{i + 1}: ch{channel_number}_s{start_sample} "
                f"- ch{channel_number}_s{end_sample}"
            ),
            "start_sample": start_sample,
            "end_sample": end_sample,
            "columns": selected_columns,
            "model_indices": model_indices,
            "data": data,
        })

    return sections


async def read_eeg_csv(
    file: UploadFile,
    row_index: int = 0,
    graph_channel: int = 1,
    section_count: int = 4,
    section_size: int | None = None,
):
    df = pd.read_csv(file.file)

    if df.empty:
        raise ValueError("CSV EEG kosong.")

    if row_index < 0 or row_index >= len(df):
        raise ValueError(
            f"row_index tidak valid. Diterima {row_index}, total row {len(df)}."
        )

    row = df.iloc[row_index]

    model_feature_columns, feature_selection_info = select_model_feature_columns(df)

    model_input = (
        row[model_feature_columns]
        .astype(float)
        .to_numpy()
        .tolist()
    )

    channel_columns = get_channel_sample_columns(
        df.columns,
        graph_channel=graph_channel,
    )

    if not channel_columns:
        raise ValueError(
            f"Tidak ditemukan kolom untuk channel ch{graph_channel}. "
            f"Pastikan format kolom seperti ch{graph_channel}_s1, "
            f"ch{graph_channel}_s2, dst."
        )

    graph_sections = build_eeg_graph_sections(
        row=row,
        channel_columns=channel_columns,
        model_feature_columns=model_feature_columns,
        section_count=section_count,
        section_size=section_size,
    )

    graph_data = []

    for section in graph_sections:
        for point in section["data"]:
            graph_data.append({
                **point,
                "section": section["name"],
            })

    return {
        "model_input": model_input,
        "graph_data": graph_data,
        "graph_sections": graph_sections,
        "feature_count": len(model_input),
        "selected_row": row_index,
        "selected_channel": graph_channel,
        "section_count": section_count,
        "section_size": section_size,
        "feature_selection_info": feature_selection_info,
        "uploaded_filename": file.filename,
    }