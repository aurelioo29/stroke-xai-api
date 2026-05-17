import re
import numpy as np
import pandas as pd
from fastapi import UploadFile

from app.utils.preprocess_eeg import get_expected_feature_count


# ============================================================
# Basic Helpers
# ============================================================

def natural_sample_sort(column_name: str):
    """
    Sort kolom EEG secara natural:
    ch1_s1, ch1_s2, ..., ch1_s1024
    """

    match = re.search(r"_s(\d+)$", str(column_name))

    if not match:
        return 0

    return int(match.group(1))


def get_channel_number(column_name: str):
    """
    Ambil nomor channel dari nama kolom.

    Contoh:
    ch1_s256 -> 1
    """

    match = re.search(r"^ch(\d+)_s\d+$", str(column_name))

    if not match:
        return "?"

    return int(match.group(1))


def get_channel_sample_columns(columns, graph_channel: int):
    """
    Ambil kolom sample EEG untuk channel tertentu.

    graph_channel = 1:
    ch1_s1 sampai ch1_s1024
    """

    pattern = re.compile(rf"^ch{graph_channel}_s\d+$")

    channel_columns = [
        col for col in columns
        if pattern.match(str(col))
    ]

    return sorted(channel_columns, key=natural_sample_sort)


def safe_subject_sort(value):
    """
    Sort subject_1, subject_2, ..., subject_10 dengan benar.
    """

    value_str = str(value)
    last_part = value_str.split("_")[-1]

    if last_part.isdigit():
        return int(last_part)

    return value_str


# ============================================================
# Model Feature Selection
# ============================================================

def select_model_feature_columns(df: pd.DataFrame):
    """
    Pilih kolom fitur yang cocok dengan scaler.

    Kemungkinan 1:
    scaler dilatih dengan trial ikut fitur:
    - drop subject + label
    - trial tetap masuk

    Kemungkinan 2:
    scaler dilatih tanpa trial:
    - drop subject + trial + label
    """

    expected_count = get_expected_feature_count()
    columns = list(df.columns)

    # Mode 1: trial tetap ikut fitur
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
        f"Cek apakah CSV, scaler, dan model berasal dari preprocessing training yang sama."
    )


# ============================================================
# EEG Section P1-P4
# ============================================================

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


def flatten_graph_sections(sections: list):
    """
    Gabungkan semua section menjadi 1 graph_data.
    """

    graph_data = []

    for section in sections:
        for point in section["data"]:
            graph_data.append({
                **point,
                "section": section["name"],
                "section_id": section["id"],
                "cycle": section["cycle"],
            })

    return graph_data


# ============================================================
# Frequency Band Analysis
# ============================================================

EEG_FREQUENCY_BANDS = [
    {
        "key": "delta",
        "label": "Delta",
        "range_label": "0.5–4 Hz",
        "low": 0.5,
        "high": 4.0,
    },
    {
        "key": "theta",
        "label": "Theta",
        "range_label": "4–8 Hz",
        "low": 4.0,
        "high": 8.0,
    },
    {
        "key": "alpha",
        "label": "Alpha",
        "range_label": "8–13 Hz",
        "low": 8.0,
        "high": 13.0,
    },
    {
        "key": "beta",
        "label": "Beta",
        "range_label": "13–30 Hz",
        "low": 13.0,
        "high": 30.0,
    },
    {
        "key": "gamma_low",
        "label": "Gamma Low",
        "range_label": "30–40 Hz",
        "low": 30.0,
        "high": 40.0,
    },
]


def get_frequency_color(percent: float):
    """
    Warna frequency power:
    Biru  = rendah / normal
    Hijau = observasi
    Kuning = early warning
    Merah = power paling dominan
    """

    value = float(percent)

    if value >= 75:
        return {
            "level": "high_power",
            "color": "#ef4444",
            "label": "High Power",
            "description": "Aktivitas frekuensi paling kuat / dominan.",
        }

    if value >= 50:
        return {
            "level": "early_warning",
            "color": "#facc15",
            "label": "Early Warning",
            "description": "Aktivitas frekuensi meningkat dan perlu diperhatikan.",
        }

    if value >= 25:
        return {
            "level": "observation",
            "color": "#22c55e",
            "label": "Observation",
            "description": "Aktivitas frekuensi sedang.",
        }

    return {
        "level": "low_power",
        "color": "#2563eb",
        "label": "Low Power",
        "description": "Aktivitas frekuensi rendah / normal.",
    }


def compute_band_power(signal_values, sampling_rate: int = 256):
    """
    Hitung power per band menggunakan FFT.

    Ini bukan CWT wavelet asli, tapi cukup untuk menghitung
    power band Delta, Theta, Alpha, Beta, Gamma.
    """

    signal = np.asarray(signal_values, dtype=np.float32)

    if signal.size == 0:
        return {band["key"]: 0.0 for band in EEG_FREQUENCY_BANDS}

    signal = signal - np.mean(signal)

    window = np.hanning(signal.size)
    windowed_signal = signal * window

    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sampling_rate)
    fft_values = np.fft.rfft(windowed_signal)
    power = np.abs(fft_values) ** 2

    band_power = {}

    for band in EEG_FREQUENCY_BANDS:
        mask = (freqs >= band["low"]) & (freqs < band["high"])

        if np.any(mask):
            band_power[band["key"]] = float(np.mean(power[mask]))
        else:
            band_power[band["key"]] = 0.0

    return band_power


# ============================================================
# Scalogram-like STFT Plot
# ============================================================

def normalize_matrix(matrix: np.ndarray):
    """
    Normalize matrix ke range 0-1.
    """

    matrix = np.asarray(matrix, dtype=np.float32)

    if matrix.size == 0:
        return matrix

    min_value = float(np.min(matrix))
    max_value = float(np.max(matrix))

    if max_value - min_value <= 1e-8:
        return np.zeros_like(matrix, dtype=np.float32)

    return (matrix - min_value) / (max_value - min_value)


def build_scalogram_plot(
    signal_values,
    sampling_rate: int = 256,
    max_frequency: float = 40.0,
    frequency_bin_count: int = 40,
    time_bin_count: int = 36,
):
    """
    Membuat scalogram-like plot tanpa scipy / pywavelets.

    Metode:
    - sinyal dibagi menjadi beberapa time window
    - tiap window dihitung FFT
    - power frekuensi 0.5–40 Hz dihitung
    - hasilnya jadi matrix frequency x time

    Output cocok untuk FE:
    {
      frequency_labels: [...],
      time_labels: [...],
      matrix: [[...]],
      highlight_region: {...}
    }
    """

    signal = np.asarray(signal_values, dtype=np.float32)

    if signal.size == 0:
        return {
            "frequency_labels": [],
            "time_labels": [],
            "matrix": [],
            "highlight_region": None,
        }

    signal = signal - np.mean(signal)

    total_samples = signal.size
    duration_seconds = total_samples / float(sampling_rate)

    if time_bin_count <= 0:
        time_bin_count = 36

    if frequency_bin_count <= 0:
        frequency_bin_count = 40

    # Agar window tidak terlalu kecil
    min_window_size = max(32, int(sampling_rate * 0.25))
    window_size = max(min_window_size, int(total_samples / min(time_bin_count, total_samples)))

    if window_size > total_samples:
        window_size = total_samples

    if time_bin_count == 1:
        centers = np.array([total_samples // 2])
    else:
        centers = np.linspace(
            window_size // 2,
            total_samples - window_size // 2,
            time_bin_count,
        ).astype(int)

    target_freqs = np.linspace(0.5, max_frequency, frequency_bin_count)

    matrix = []

    for center in centers:
        start = int(center - window_size // 2)
        end = int(start + window_size)

        if start < 0:
            start = 0
            end = window_size

        if end > total_samples:
            end = total_samples
            start = max(0, end - window_size)

        segment = signal[start:end]

        if segment.size < 4:
            segment_power = np.zeros_like(target_freqs, dtype=np.float32)
            matrix.append(segment_power)
            continue

        segment = segment - np.mean(segment)

        window = np.hanning(segment.size)
        segment = segment * window

        freqs = np.fft.rfftfreq(segment.size, d=1.0 / sampling_rate)
        fft_values = np.fft.rfft(segment)
        power = np.abs(fft_values) ** 2

        # Interpolasi power ke target frequency grid
        interp_power = np.interp(
            target_freqs,
            freqs,
            power,
            left=0.0,
            right=0.0,
        )

        matrix.append(interp_power)

    # matrix sekarang time x frequency
    matrix = np.asarray(matrix, dtype=np.float32).T

    # Smooth ringan supaya lebih mirip heatmap/scalogram
    matrix = smooth_matrix(matrix)

    normalized = normalize_matrix(matrix)

    # Untuk FE, frequency axis dari tinggi ke rendah:
    # 40Hz di atas, 0.5Hz di bawah
    normalized = np.flipud(normalized)

    freq_labels_full = np.flipud(target_freqs)

    # Supaya FE tidak kebanyakan label, kita tetap kirim semua row,
    # label-nya dibuat ringkas.
    frequency_labels = [
        f"{freq:.1f}Hz" if freq < 10 else f"{freq:.0f}Hz"
        for freq in freq_labels_full
    ]

    time_positions = np.linspace(0, duration_seconds, time_bin_count)

    time_labels = [
        f"{t:.1f}s" if duration_seconds <= 10 else f"{t:.0f}s"
        for t in time_positions
    ]

    highlight_region = find_scalogram_highlight_region(normalized)

    return {
        "frequency_labels": frequency_labels,
        "time_labels": time_labels,
        "matrix": normalized.round(4).tolist(),
        "highlight_region": highlight_region,
        "duration_seconds": round(float(duration_seconds), 4),
        "sampling_rate": sampling_rate,
        "method": "stft_like_fft",
        "note": (
            "This is a scalogram-style time-frequency plot generated using "
            "windowed FFT. For true wavelet scalogram, use CWT/PyWavelets later."
        ),
    }


def smooth_matrix(matrix: np.ndarray):
    """
    Smooth sederhana tanpa scipy.
    Biar heatmap tidak terlalu kotak-kotak brutal.
    """

    matrix = np.asarray(matrix, dtype=np.float32)

    if matrix.ndim != 2:
        return matrix

    rows, cols = matrix.shape
    result = matrix.copy()

    for r in range(rows):
        for c in range(cols):
            r1 = max(0, r - 1)
            r2 = min(rows, r + 2)
            c1 = max(0, c - 1)
            c2 = min(cols, c + 2)

            result[r, c] = np.mean(matrix[r1:r2, c1:c2])

    return result


def find_scalogram_highlight_region(normalized_matrix: np.ndarray):
    """
    Cari area paling dominan untuk dibuat highlight box di FE.

    Output format cocok dengan EEGScalogramChart:
    {
      xStart, xEnd, yStart, yEnd
    }
    """

    matrix = np.asarray(normalized_matrix, dtype=np.float32)

    if matrix.size == 0:
        return None

    rows, cols = matrix.shape

    max_index = np.unravel_index(np.argmax(matrix), matrix.shape)
    max_row, max_col = int(max_index[0]), int(max_index[1])

    # Box sekitar hotspot
    x_radius = max(1, cols // 12)
    y_radius = max(2, rows // 10)

    x_start = max(0, max_col - x_radius)
    x_end = min(cols - 1, max_col + x_radius)

    y_start = max(0, max_row - y_radius)
    y_end = min(rows - 1, max_row + y_radius)

    return {
        "xStart": int(x_start),
        "xEnd": int(x_end),
        "yStart": int(y_start),
        "yEnd": int(y_end),
        "peakRow": int(max_row),
        "peakCol": int(max_col),
        "peakValue": round(float(matrix[max_row, max_col]), 4),
    }


def build_frequency_analysis(
    graph_sections: list,
    signal_values=None,
    sampling_rate: int = 256,
):
    """
    Membuat:
    - band power heatmap P1-P4
    - dominant_frequency
    - scalogram_plot untuk FE bawah seperti konsep DPL
    """

    section_band_rows = []
    all_power_values = []

    for section in graph_sections:
        section_signal_values = [
            point["value"]
            for point in section.get("data", [])
        ]

        band_power = compute_band_power(
            signal_values=section_signal_values,
            sampling_rate=sampling_rate,
        )

        for band in EEG_FREQUENCY_BANDS:
            power_value = float(band_power.get(band["key"], 0.0))
            all_power_values.append(power_value)

        section_band_rows.append({
            "section_id": section["id"],
            "section_name": section["name"],
            "section_display_name": section.get("display_name", section["name"]),
            "start_sample": section["start_sample"],
            "end_sample": section["end_sample"],
            "bands": band_power,
        })

    max_power = max(all_power_values) if all_power_values else 0.0

    heatmap_cells = []
    dominant_cell = None

    for row in section_band_rows:
        for band in EEG_FREQUENCY_BANDS:
            raw_power = float(row["bands"].get(band["key"], 0.0))

            if max_power > 0:
                power_percent = raw_power / max_power * 100
            else:
                power_percent = 0.0

            color_info = get_frequency_color(power_percent)

            cell = {
                "section_id": row["section_id"],
                "section_name": row["section_name"],
                "section_display_name": row["section_display_name"],
                "start_sample": row["start_sample"],
                "end_sample": row["end_sample"],

                "band_key": band["key"],
                "band_label": band["label"],
                "range_label": band["range_label"],
                "low": band["low"],
                "high": band["high"],

                "raw_power": round(raw_power, 6),
                "power_percent": round(float(power_percent), 2),

                "level": color_info["level"],
                "color": color_info["color"],
                "label": color_info["label"],
                "description": color_info["description"],
            }

            heatmap_cells.append(cell)

            if dominant_cell is None or cell["power_percent"] > dominant_cell["power_percent"]:
                dominant_cell = cell

    if signal_values is None:
        signal_values = []

        for section in graph_sections:
            for point in section.get("data", []):
                signal_values.append(point["value"])

    scalogram_plot = build_scalogram_plot(
        signal_values=signal_values,
        sampling_rate=sampling_rate,
        max_frequency=40.0,
        frequency_bin_count=40,
        time_bin_count=36,
    )

    if dominant_cell:
        explanation = (
            f"Aktivitas frekuensi paling dominan terdeteksi pada band "
            f"{dominant_cell['band_label']} ({dominant_cell['range_label']}) "
            f"di section {dominant_cell['section_name']} "
            f"(sample s{dominant_cell['start_sample']}–s{dominant_cell['end_sample']}). "
            f"Warna merah menunjukkan power frekuensi tertinggi, sedangkan biru "
            f"menunjukkan power rendah."
        )
    else:
        explanation = "Frequency analysis tidak menemukan aktivitas dominan."

    return {
        "sampling_rate": sampling_rate,
        "bands": EEG_FREQUENCY_BANDS,
        "sections": [
            {
                "section_id": row["section_id"],
                "section_name": row["section_name"],
                "section_display_name": row["section_display_name"],
                "start_sample": row["start_sample"],
                "end_sample": row["end_sample"],
            }
            for row in section_band_rows
        ],
        "heatmap_cells": heatmap_cells,
        "dominant_frequency": dominant_cell,
        "explanation": explanation,
        "scalogram_plot": scalogram_plot,
        "color_legend": [
            {
                "color": "#2563eb",
                "label": "Low / Normal",
                "description": "Power frekuensi rendah.",
            },
            {
                "color": "#22c55e",
                "label": "Observation",
                "description": "Power frekuensi sedang.",
            },
            {
                "color": "#facc15",
                "label": "Early Warning",
                "description": "Power frekuensi mulai meningkat.",
            },
            {
                "color": "#ef4444",
                "label": "High Power",
                "description": "Power frekuensi paling tinggi / dominan.",
            },
        ],
    }


# ============================================================
# Subject Graphs
# ============================================================

def build_subject_average_graphs(
    df: pd.DataFrame,
    channel_columns: list,
    graph_channel: int = 1,
    section_count: int = 4,
    section_size: int = 256,
    sampling_rate: int = 256,
):
    """
    Untuk setiap subject:
    - ambil semua trial
    - rata-rata sinyal channel tersebut
    - buat graph P1-P4
    - buat frequency_analysis subject tersebut
    - buat scalogram_plot subject tersebut
    """

    if "subject" not in df.columns:
        return []

    subject_graphs = []

    subjects = sorted(
        df["subject"].dropna().unique().tolist(),
        key=safe_subject_sort,
    )

    for subject in subjects:
        subject_df = df[df["subject"] == subject]

        if subject_df.empty:
            continue

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

        graph_data = flatten_graph_sections(sections)

        signal_values = averaged_signal.astype(float).to_numpy().tolist()

        frequency_analysis = build_frequency_analysis(
            graph_sections=sections,
            signal_values=signal_values,
            sampling_rate=sampling_rate,
        )

        subject_graphs.append({
            "subject": subject,
            "trial_count": int(len(subject_df)),
            "selected_channel": graph_channel,

            "graph_data": graph_data,
            "graph_sections": sections,

            "frequency_analysis": frequency_analysis,
        })

    return subject_graphs


# ============================================================
# Main CSV Reader
# ============================================================

async def read_eeg_csv(
    file: UploadFile,
    row_index: int = 0,
    graph_channel: int = 1,
    section_count: int = 4,
    section_size: int = 256,
    cycle_count: int = 1,
    sampling_rate: int = 256,
):
    """
    Baca CSV EEG dan return:
    - model_input
    - selected row graph
    - selected row frequency_analysis + scalogram_plot
    - subject_graphs, masing-masing punya frequency_analysis + scalogram_plot
    """

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

    graph_data = flatten_graph_sections(graph_sections)

    selected_signal_values = (
        row[channel_columns]
        .astype(float)
        .to_numpy()
        .tolist()
    )

    frequency_analysis = build_frequency_analysis(
        graph_sections=graph_sections,
        signal_values=selected_signal_values,
        sampling_rate=sampling_rate,
    )

    subject_graphs = build_subject_average_graphs(
        df=df,
        channel_columns=channel_columns,
        graph_channel=graph_channel,
        section_count=section_count,
        section_size=section_size,
        sampling_rate=sampling_rate,
    )

    return {
        "model_input": model_input,

        # selected row/trial graph
        "graph_data": graph_data,
        "graph_sections": graph_sections,

        # selected row/trial frequency
        "frequency_analysis": frequency_analysis,
        "sampling_rate": sampling_rate,

        # subject graphs
        "subject_graphs": subject_graphs,
        "subject_count": len(subject_graphs),

        "feature_count": len(model_input),
        "selected_row": row_index,
        "selected_channel": graph_channel,

        "section_count": section_count,
        "section_size": section_size,
        "cycle_count": cycle_count,

        "feature_selection_info": feature_selection_info,
        "uploaded_filename": file.filename,
    }