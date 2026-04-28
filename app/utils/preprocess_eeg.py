import os
import joblib
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

SCALER_PATH = os.path.join(BASE_DIR, "app", "models", "eeg_scaler.pkl")
PCA_PATH = os.path.join(BASE_DIR, "app", "models", "eeg_pca.pkl")


scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)


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

    if features.shape[1] != 16385:
        raise ValueError(
            f"Jumlah fitur EEG harus 16385, tetapi diterima {features.shape[1]}"
        )

    X_scaled = scaler.transform(features)
    X_pca = pca.transform(X_scaled)
    X_fuzzy = fuzzy_transform(X_pca)

    if X_fuzzy.shape[1] != 1500:
        raise ValueError(
            f"Hasil fuzzy harus 1500 fitur, tetapi diterima {X_fuzzy.shape[1]}"
        )

    return X_fuzzy.reshape(1, 1500, 1).astype(np.float32)


def preprocess_eeg_array(eeg_array):
    eeg_array = np.array(eeg_array, dtype=np.float32)

    if eeg_array.ndim == 1 and len(eeg_array) == 1500:
        return eeg_array.reshape(1, 1500, 1).astype(np.float32)

    if eeg_array.ndim == 1 and len(eeg_array) == 16385:
        return preprocess_eeg_features(eeg_array)

    raise ValueError(
        f"Format EEG tidak valid. Diterima shape {eeg_array.shape}"
    )