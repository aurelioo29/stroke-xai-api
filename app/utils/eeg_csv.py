import re
import pandas as pd
from fastapi import UploadFile

from app.utils.preprocess_eeg import get_expected_feature_count


def natural_sample_sort(column_name: str):
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

    return sorted(channel_columns, key=natural_sample_sort)


def select_model_feature_columns(df: pd.DataFrame):
    expected_count = get_expected_feature_count()
    columns = list(df.columns)

    # Mode 1: trial ikut fitur
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

    # Mode 2: trial tidak ikut fitur
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
        f"Cek apakah CSV, scaler, dan model berasal dari preprocessing training yang sama."
    )


def build_eeg_graph_sections(
    row: pd.Series,
    channel_columns: list,
    model_feature_columns: list | None = None,
    section_count: int = 4,
    section_size: int = 256,
    cycle_count: int = 1,
):
    """
    Final section:
    P1 = s1-s256
    P2 = s257-s512
    P3 = s513-s768
    P4 = s769-s1024
    """

    total_samples = len(channel_columns)

    if total_samples == 0:
        return []

    if section_count <= 0:
        section_count = 4

    if section_size is None or section_size <= 0:
        section_size = 256

    if cycle_count is None or cycle_count <= 0:
        cycle_count = 1

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

            model_indices = []

            if model_feature_columns:
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
                "display_name": section_name if cycle_count == 1 else f"C{cycle_number} • {section_name}",
                "title": (
                    f"{section_name}: "
                    f"ch{channel_number}_s{start_sample} - "
                    f"ch{channel_number}_s{end_sample}"
                ),
                "start_sample": start_sample,
                "end_sample": end_sample,
                "channel": channel_number,
                "columns": selected_columns,
                "model_indices": model_indices,
                "data": data,
            })

    return sections


def build_subject_average_graphs(
    df: pd.DataFrame,
    channel_columns: list,
    graph_channel: int = 1,
    section_count: int = 4,
    section_size: int = 256,
):
    """
    Membuat 1 grafik rata-rata untuk setiap subject.

    Jadi:
    subject_1 = rata-rata 50 trial
    subject_2 = rata-rata 50 trial
    dst.
    """

    if "subject" not in df.columns:
        return []

    subject_graphs = []

    subjects = sorted(
        df["subject"].dropna().unique().tolist(),
        key=lambda value: int(str(value).split("_")[-1])
        if str(value).split("_")[-1].isdigit()
        else str(value)
    )

    for subject in subjects:
        subject_df = df[df["subject"] == subject]

        if subject_df.empty:
            continue

        # Rata-rata 50 trial / semua trial milik subject
        averaged_signal = subject_df[channel_columns].astype(float).mean(axis=0)

        averaged_row = pd.Series(
            data=averaged_signal.values,
            index=channel_columns,
        )

        sections = build_eeg_graph_sections(
            row=averaged_row,
            channel_columns=channel_columns,
            model_feature_columns=None,
            section_count=section_count,
            section_size=section_size,
            cycle_count=1,
        )

        graph_data = []

        for section in sections:
            for point in section["data"]:
                graph_data.append({
                    **point,
                    "section": section["name"],
                    "section_id": section["id"],
                    "cycle": section["cycle"],
                })

        subject_graphs.append({
            "subject": subject,
            "trial_count": int(len(subject_df)),
            "selected_channel": graph_channel,
            "graph_data": graph_data,
            "graph_sections": sections,
        })

    return subject_graphs


async def read_eeg_csv(
    file: UploadFile,
    row_index: int = 0,
    graph_channel: int = 1,
    section_count: int = 4,
    section_size: int = 256,
    cycle_count: int = 1,
):
    df = pd.read_csv(file.file)

    if df.empty:
        raise ValueError("CSV EEG kosong.")

    if row_index < 0 or row_index >= len(df):
        raise ValueError(
            f"row_index tidak valid. "
            f"Diterima {row_index}, total row {len(df)}."
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
            f"ch{graph_channel}_s2, ..., ch{graph_channel}_s1024."
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

    subject_graphs = build_subject_average_graphs(
        df=df,
        channel_columns=channel_columns,
        graph_channel=graph_channel,
        section_count=section_count,
        section_size=section_size,
    )

    subject_count = len(subject_graphs)

    return {
        "model_input": model_input,

        # Grafik selected row / trial
        "graph_data": graph_data,
        "graph_sections": graph_sections,

        # Grafik rata-rata setiap subject
        "subject_graphs": subject_graphs,
        "subject_count": subject_count,

        "feature_count": len(model_input),
        "selected_row": row_index,
        "selected_channel": graph_channel,

        "section_count": section_count,
        "section_size": section_size,
        "cycle_count": cycle_count,

        "feature_selection_info": feature_selection_info,
        "uploaded_filename": file.filename,
    }