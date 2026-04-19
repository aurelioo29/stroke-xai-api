import requests
import json
import numpy as np

API_URL = "http://127.0.0.1:8000/predict/eeg-xai"

def main():
    eeg_signal = np.random.rand(1500).astype("float32").tolist()

    response = requests.post(API_URL, json=eeg_signal)

    print(f"Status Code: {response.status_code}")

    try:
        result = response.json()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception:
        print("Response bukan JSON:")
        print(response.text)

if __name__ == "__main__":
    main()