import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b")

print("OPENROUTER_API_KEY loaded:", bool(api_key))
print("OPENROUTER_MODEL:", model_name)

if not api_key:
    raise ValueError("OPENROUTER_API_KEY tidak ditemukan di .env")

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    # optional
    "HTTP-Referer": "http://localhost:3000",
    "X-Title": "stroke-xai-api-test",
}

payload = {
    "model": model_name,
    "messages": [
        {"role": "user", "content": "Balas singkat: koneksi OpenRouter berhasil."}
    ],
    "temperature": 0.2,
}

try:
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    print("Status Code:", response.status_code)

    data = response.json()
    if response.ok:
        print("=== SUCCESS ===")
        print(data["choices"][0]["message"]["content"])
    else:
        print("=== ERROR ===")
        print(data)
except Exception as e:
    print("=== ERROR ===")
    print(type(e).__name__)
    print(str(e))