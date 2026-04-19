import numpy as np


def preprocess_eeg_array(eeg_array):
    eeg_array = np.array(eeg_array, dtype=np.float32)

    if len(eeg_array) != 1500:
        raise ValueError("EEG data harus panjang 1500")

    eeg_array = eeg_array.reshape(1, 1500, 1)
    return eeg_array