import requests
from app.config import GROQ_API_KEY, GROQ_LLM_MODEL, USE_HF_FALLBACK, HF_TOKEN, HF_GRAMMAR_MODEL, REQUEST_TIMEOUT

# Groq completions endpoint (may vary: completions or chat)
GROQ_COMPLETIONS_URL = "https://api.groq.com/openai/v1/completions"

def correct_with_groq_llm(text: str) -> str:
    if not GROQ_API_KEY:
        raise Exception("Missing GROQ_API_KEY for LLM")
    # Prompt: minimal, ask to return only corrected text
    prompt = f"Correct the grammar and punctuation of the following text. Return only the corrected text (no commentary):\n\nText: \"{text}\"\n\nCorrected:"
    payload = {
        "model": GROQ_LLM_MODEL,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.0
    }
    try:
        r = requests.post(GROQ_COMPLETIONS_URL, headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }, json=payload, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        raise Exception(f"Groq LLM network error: {e}")

    if r.status_code != 200:
        raise Exception(f"Groq LLM Error ({r.status_code}): {r.text}")

    try:
        j = r.json()
    except Exception:
        raise Exception(f"Groq LLM: invalid JSON: {r.text}")

    # common completions output: {"choices":[{"text":"..."}]}
    if isinstance(j, dict):
        if "choices" in j and j["choices"]:
            txt = j["choices"][0].get("text") or j["choices"][0].get("message", {}).get("content")
            if txt:
                return txt.strip()
        # fallback keys
        for k in ("output_text", "text"):
            if k in j:
                return str(j[k]).strip()
    return str(j)

def correct_with_hf_router(text: str) -> str:
    if not HF_TOKEN:
        raise Exception("Missing HF_TOKEN for HF fallback")
    url = f"https://router.huggingface.co/models/{HF_GRAMMAR_MODEL}?use_cache=false"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Accept": "application/json", "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers, json={"inputs": text}, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        raise Exception(f"HF router network error: {e}")

    if r.status_code != 200:
        raise Exception(f"HF Grammar Error ({r.status_code}): {r.text}")

    try:
        data = r.json()
    except Exception:
        raise Exception(f"HF Grammar: invalid JSON: {r.text}")

    # multiple formats
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip()
    if isinstance(data, dict) and "choices" in data and data["choices"]:
        return data["choices"][0].get("text", "").strip()
    return str(data)

def correct_grammar(text: str) -> str:
    """
    Primary: Groq LLM.
    Fallback: HuggingFace router if enabled.
    Final fallback: return original text.
    """
    try:
        corrected = correct_with_groq_llm(text)
        if corrected and corrected.strip():
            return corrected
    except Exception as e:
        # try HF fallback only if configured
        if USE_HF_FALLBACK:
            try:
                corrected = correct_with_hf_router(text)
                if corrected and corrected.strip():
                    return corrected
            except Exception as hf_e:
                # raise the original or combined error for debugging
                raise Exception(f"Groq LLM failed: {e}; HF fallback failed: {hf_e}")
        # otherwise fall through and return original
        print("Grammar correction warning:", e)
    return text
