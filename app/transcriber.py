import requests
from app.config import GROQ_API_KEY, GROQ_ASR_MODEL, REQUEST_TIMEOUT

GROQ_ASR_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

def transcribe_bytes(audio_bytes: bytes) -> str:
    """
    Send audio bytes to Groq Whisper ASR and return the transcript text.
    Raises Exception on non-200 responses with helpful messages.
    """
    if not GROQ_API_KEY:
        raise Exception("Missing GROQ_API_KEY in environment")

    files = {
        "file": ("audio.wav", audio_bytes, "audio/wav")
    }
    data = {"model": GROQ_ASR_MODEL}

    try:
        r = requests.post(GROQ_ASR_URL, headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, files=files, data=data, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        raise Exception(f"Groq ASR network error: {e}")

    if r.status_code != 200:
        # include body in error for debugging
        raise Exception(f"Groq ASR Error ({r.status_code}): {r.text}")

    try:
        payload = r.json()
    except Exception:
        raise Exception(f"Groq ASR: invalid JSON response: {r.text}")

    # expected response: {"text": "..."}
    text = payload.get("text") or payload.get("transcription") or payload.get("result")
    if not text:
        # fallback: return stringified json
        return str(payload)
    return text
