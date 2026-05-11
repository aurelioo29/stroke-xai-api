import os
import joblib
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

SCALER_PATH = os.path.join(BASE_DIR, "app", "models", "eeg_scaler.pkl")
PCA_PATH = os.path.join(BASE_DIR, "app", "models", "eeg_pca.pkl")


scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)


def get_expected_feature_count():
    """
    Jumlah fitur yang diharapkan harus mengikuti scaler,
    bukan hardcode 16385 / 16384.

    Kalau scaler dulu dilatih dengan trial ikut fitur:
    expected = 16385

    Kalau scaler baru dilatih tanpa trial:
    expected = 16384
    """

    if hasattr(scaler, "n_features_in_"):
        return int(scaler.n_features_in_)

    raise ValueError(
        "Scaler tidak memiliki attribute n_features_in_. "
        "Pastikan scaler disimpan dari StandardScaler yang sudah fit."
    )


def get_expected_fuzzy_count():
    """
    Jumlah fitur fuzzy = jumlah komponen PCA x 3 membership:
    low, medium, high.
    """

    if hasattr(pca, "n_components_"):
        return int(pca.n_components_) * 3

    if hasattr(pca, "n_components"):
        return int(pca.n_components) * 3

    return 1500


def fuzzy_transform(X):
    low = np.exp(-((X - (-1)) ** 2) / (2 * 0.5 ** 2))
    medium = np.exp(-((X - 0) ** 2) / (2 * 0.5 ** 2))
    high = np.exp(-((X - 1) ** 2) / (2 * 0.5 ** 2))

    X_fuzzy = np.concatenate(
        [low, medium, high],
        axis=1,
    )

    return X_fuzzy


def preprocess_eeg_features(features):
    features = np.array(features, dtype=np.float32)

    if features.ndim == 1:
        features = features.reshape(1, -1)

    expected_feature_count = get_expected_feature_count()

    if features.shape[1] != expected_feature_count:
        raise ValueError(
            f"Jumlah fitur EEG tidak sesuai dengan scaler. "
            f"Scaler membutuhkan {expected_feature_count} fitur, "
            f"tetapi diterima {features.shape[1]} fitur. "
            f"Pastikan kolom CSV yang dikirim sama dengan preprocessing saat training."
        )

    X_scaled = scaler.transform(features)
    X_pca = pca.transform(X_scaled)
    X_fuzzy = fuzzy_transform(X_pca)

    expected_fuzzy_count = get_expected_fuzzy_count()

    if X_fuzzy.shape[1] != expected_fuzzy_count:
        raise ValueError(
            f"Hasil fuzzy tidak sesuai. "
            f"Expected {expected_fuzzy_count} fitur, "
            f"tetapi diterima {X_fuzzy.shape[1]} fitur."
        )

    return X_fuzzy.reshape(1, expected_fuzzy_count, 1).astype(np.float32)


def preprocess_eeg_array(eeg_array):
    eeg_array = np.array(eeg_array, dtype=np.float32)

    expected_feature_count = get_expected_feature_count()
    expected_fuzzy_count = get_expected_fuzzy_count()

    # Jika input sudah hasil fuzzy final, misalnya 1500
    if eeg_array.ndim == 1 and len(eeg_array) == expected_fuzzy_count:
        return eeg_array.reshape(1, expected_fuzzy_count, 1).astype(np.float32)

    # Jika input masih fitur mentah dari CSV
    if eeg_array.ndim == 1 and len(eeg_array) == expected_feature_count:
        return preprocess_eeg_features(eeg_array)

    raise ValueError(
        f"Format EEG tidak valid. Diterima shape {eeg_array.shape}. "
        f"Format yang diterima: "
        f"{expected_feature_count} fitur mentah atau {expected_fuzzy_count} fitur fuzzy."
    )