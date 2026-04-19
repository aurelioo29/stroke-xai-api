import requests
import json
import numpy as np

API_URL = "http://127.0.0.1:8000/predict/eeg"

def main():
    # generate dummy EEG 1500 data point
    eeg_signal = np.random.rand(1500).astype("float32").tolist()

    response = requests.post(API_URL, json=eeg_signal)

    print(f"Status Code: {response.status_code}")

    try:
        data = response.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        print("Response bukan JSON:")
        print(response.text)

if __name__ == "__main__":
    main()