import requests
from app.config import GROQ_API_KEY, GROQ_ASR_MODEL, REQUEST_TIMEOUT

GROQ_ASR_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

def transcribe_bytes_from_bytes(audio_bytes: bytes) -> str:
    if not GROQ_API_KEY:
        raise Exception("Missing GROQ_API_KEY in environment")

    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {"model": GROQ_ASR_MODEL}

    try:
        r = requests.post(
            GROQ_ASR_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as e:
        raise Exception(f"Groq ASR network error: {e}")

    if r.status_code != 200:
        raise Exception(f"Groq ASR Error ({r.status_code}): {r.text}")

    try:
        payload = r.json()
    except:
        raise Exception(f"Groq ASR: invalid JSON response: {r.text}")

    text = (
        payload.get("text")
        or payload.get("transcription")
        or payload.get("result")
    )
    return text or str(payload)


def transcribe_from_path(path: str) -> str:
    with open(path, "rb") as f:
        return transcribe_bytes_from_bytes(f.read())
