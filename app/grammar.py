import requests
from app.config import GROQ_API_KEY, GROQ_LLM_MODEL, USE_HF_FALLBACK, HF_TOKEN, HF_GRAMMAR_MODEL, REQUEST_TIMEOUT

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

def correct_with_groq_llm(text: str) -> str:
    if not GROQ_API_KEY:
        raise Exception("Missing GROQ_API_KEY for LLM")

    payload = {
        "model": GROQ_LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a grammar correction assistant. Return ONLY the corrected sentence, nothing else."},
            {"role": "user", "content": text}
        ],
        "temperature": 0.0,
        "max_tokens": 256
    }

    try:
        r = requests.post(
            GROQ_CHAT_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as e:
        raise Exception(f"Groq LLM network error: {e}")

    if r.status_code != 200:
        raise Exception(f"Groq LLM Error ({r.status_code}): {r.text}")

    try:
        j = r.json()
        # Extract chat completion text
        corrected = j["choices"][0]["message"]["content"].strip()
        return corrected
    except Exception:
        raise Exception(f"Groq LLM invalid JSON: {r.text}")


def correct_with_hf_router(text: str) -> str:
    if not HF_TOKEN:
        raise Exception("Missing HF_TOKEN for HF fallback")

    url = f"https://router.huggingface.co/models/{HF_GRAMMAR_MODEL}?use_cache=false"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(url, headers=headers, json={"inputs": text}, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        raise Exception(f"HF router network error: {e}")

    if r.status_code != 200:
        raise Exception(f"HF Grammar Error ({r.status_code}): {r.text}")

    try:
        data = r.json()
    except Exception:
        raise Exception(f"HF Grammar invalid JSON: {r.text}")

    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip()
    if isinstance(data, dict) and "choices" in data and data["choices"]:
        return data["choices"][0].get("text", "").strip()

    return str(data)


def correct_grammar(text: str) -> str:
    try:
        corrected = correct_with_groq_llm(text)
        if corrected.strip():
            return corrected
    except Exception as e:
        if USE_HF_FALLBACK:
            try:
                corrected = correct_with_hf_router(text)
                if corrected.strip():
                    return corrected
            except Exception as hf_e:
                raise Exception(f"Groq LLM failed: {e}; HF failed: {hf_e}")

        print("Grammar correction warning:", e)

    return text
