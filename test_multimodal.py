import requests
import json
import numpy as np
from pathlib import Path

API_URL = "http://127.0.0.1:8000/predict/multimodal"
IMAGE_PATH = "file_mri.jpg"

def main():
    image_file = Path(IMAGE_PATH)

    if not image_file.exists():
        print(f"File tidak ditemukan: {IMAGE_PATH}")
        return

    eeg_signal = np.random.rand(1500).astype("float32").tolist()

    with open(image_file, "rb") as f:
        files = {
            "file": (image_file.name, f, "image/jpeg")
        }

        data = {
            "eeg_json": json.dumps(eeg_signal)
        }

        response = requests.post(API_URL, files=files, data=data)

    print(f"Status Code: {response.status_code}")

    try:
        result = response.json()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception:
        print("Response bukan JSON:")
        print(response.text)

if __name__ == "__main__":
    main()