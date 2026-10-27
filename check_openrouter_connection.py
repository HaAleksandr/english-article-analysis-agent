
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("ERROR: API key was not loaded from .env")
    raise SystemExit(1)

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

payload = {
    "model": "liquid/lfm-2.5-2.6b:free",
    "messages": [
        {
            "role": "user",
            "content": "Classify the article in 'a dog'. Reply briefly.",
        }
    ],
}

try:
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=30,
    )

    print("HTTP status:", response.status_code)
    print("Response body:")
    print(response.text[:3000])

except requests.RequestException as exc:
    print("Request failed:", str(exc))
