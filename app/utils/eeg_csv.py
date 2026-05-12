import re
import pandas as pd
import numpy as np
from fastapi import UploadFile

from app.utils.preprocess_eeg import get_expected_feature_count


def natural_sample_sort(column_name: str):
    """
    Sort kolom seperti ch1_s1, ch1_s2, ..., ch1_s1000.
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
    """
    Ambil kolom channel tertentu.
    Contoh graph_channel=1:
    ch1_s1, ch1_s2, ch1_s3, ...
    """

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
    Pilih kolom fitur yang cocok dengan scaler.

    Jika scaler lama dilatih dengan trial ikut fitur:
    - drop subject + label
    - trial tetap masuk

    Jika scaler baru dilatih tanpa trial:
    - drop subject + trial + label
    """

    expected_count = get_expected_feature_count()
    columns = list(df.columns)

    # Mode 1: trial tetap ikut
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

    # Mode 2: trial dibuang
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
        f"Cek apakah CSV, scaler, dan model berasal dari preprocessing yang sama."
    )


def build_eeg_graph_sections(
    row: pd.Series,
    channel_columns: list,
    model_feature_columns: list,
    section_count: int = 4,
    section_size: int = 20,
    cycle_count: int = 2,
):
    """
    Membuat section EEG.

    Default:
    section_count = 4
    section_size = 20
    cycle_count = 2

    Hasil:
    Cycle 1:
    P1 = s1-s20
    P2 = s21-s40
    P3 = s41-s60
    P4 = s61-s80

    Cycle 2:
    P1 = s81-s100
    P2 = s101-s120
    P3 = s121-s140
    P4 = s141-s160
    """

    total_samples = len(channel_columns)

    if total_samples == 0:
        return []

    if section_count <= 0:
        section_count = 4

    if section_size is None or section_size <= 0:
        section_size = 20

    # cycle_count = 0 berarti auto sampai sample habis
    if cycle_count is None or cycle_count <= 0:
        samples_per_cycle = section_count * section_size
        cycle_count = int(np.ceil(total_samples / samples_per_cycle))

    sections = []

    for cycle_index in range(cycle_count):
        cycle_offset = cycle_index * section_count * section_size

        for section_index in range(section_count):
            start_idx = cycle_offset + section_index * section_size
            end_idx = min(start_idx + section_size, total_samples)

            if start_idx >= total_samples:
                break

            selected_columns = channel_columns[start_idx:end_idx]

            if not selected_columns:
                continue

            start_sample = natural_sample_sort(selected_columns[0])
            end_sample = natural_sample_sort(selected_columns[-1])
            channel_number = get_channel_number(selected_columns[0])

            section_name = f"P{section_index + 1}"
            cycle_number = cycle_index + 1
            section_id = f"C{cycle_number}_P{section_index + 1}"

            model_indices = [
                model_feature_columns.index(col)
                for col in selected_columns
                if col in model_feature_columns
            ]

            data = [
                {
                    "sample": natural_sample_sort(col),
                    "value": float(row[col]),
                    "section_id": section_id,
                    "section": section_name,
                    "cycle": cycle_number,
                }
                for col in selected_columns
            ]

            sections.append({
                "id": section_id,
                "name": section_name,
                "cycle": cycle_number,
                "cycle_label": f"Cycle {cycle_number}",
                "display_name": f"C{cycle_number} • {section_name}",
                "title": (
                    f"C{cycle_number} {section_name}: "
                    f"ch{channel_number}_s{start_sample} - "
                    f"ch{channel_number}_s{end_sample}"
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
    section_size: int = 20,
    cycle_count: int = 2,
):
    """
    Baca CSV EEG dan return:
    - model_input
    - graph_data
    - graph_sections
    """

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
        cycle_count=cycle_count,
    )

    graph_data = []

    for section in graph_sections:
        for point in section["data"]:
            graph_data.append({
                **point,
                "section": section["name"],
                "section_id": section["id"],
                "cycle": section["cycle"],
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
        "cycle_count": cycle_count,
        "feature_selection_info": feature_selection_info,
        "uploaded_filename": file.filename,
    }