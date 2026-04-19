import requests
import json
from pathlib import Path

API_URL = "http://127.0.0.1:8000/predict/mri"

# ganti dengan path file MRI kamu
IMAGE_PATH = "file_mri.jpg"

def main():
    image_file = Path(IMAGE_PATH)

    if not image_file.exists():
        print(f"File tidak ditemukan: {IMAGE_PATH}")
        return

    with open(image_file, "rb") as f:
        files = {
            "file": (image_file.name, f, "image/jpeg")
        }

        response = requests.post(API_URL, files=files)

    print(f"Status Code: {response.status_code}")

    try:
        data = response.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        print("Response bukan JSON:")
        print(response.text)

if __name__ == "__main__":
    main()