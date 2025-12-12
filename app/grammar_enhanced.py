"""
Enhanced grammar correction with LanguageTool (offline, free, rule-based).
Priority: LanguageTool (local) → HF transformer (local) → Groq API (limited)
"""
import logging
from app.config import (
    GROQ_API_KEY, GROQ_LLM_MODEL, REQUEST_TIMEOUT,
    USE_HF_FALLBACK, HF_TOKEN, HF_GRAMMAR_MODEL, USE_LOCAL_LANGUAGE_TOOL
)

logger = logging.getLogger(__name__)


def correct_with_language_tool(text: str) -> str:
    """
    Correct grammar using LanguageTool (offline, rule-based, free).
    No API keys needed. Fast and reliable.
    """
    try:
        import language_tool_python
    except ImportError:
        logger.error("language_tool_python not installed. Run: pip install language-tool-python")
        raise ImportError("Install language-tool-python: pip install language-tool-python")
    
    try:
        tool = language_tool_python.LanguageTool('en-US')
        matches = tool.check(text)
        corrected = language_tool_python.utils.correct(text, matches)
        logger.info(f"LanguageTool corrected {len(matches)} issues")
        return corrected
    except Exception as e:
        logger.error(f"LanguageTool failed: {e}")
        raise


def correct_with_hf_transformer(text: str) -> str:
    """
    Correct grammar using Hugging Face FLAN-T5 (local, transformer-based).
    More advanced than rule-based, but slower. Runs on CPU.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers torch")
        raise ImportError("Install transformers: pip install transformers torch")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_GRAMMAR_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(HF_GRAMMAR_MODEL)
        
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(**inputs, max_length=512, num_beams=4)
        corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"HF FLAN-T5 corrected grammar")
        return corrected
    except Exception as e:
        logger.error(f"HF transformer failed: {e}")
        raise


def correct_with_groq_llm(text: str) -> str:
    """Correct grammar using Groq Llama API (fallback, limited quota)."""
    import requests
    
    if not GROQ_API_KEY:
        raise Exception("Missing GROQ_API_KEY for LLM")

    GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
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
        corrected = j["choices"][0]["message"]["content"].strip()
        return corrected
    except Exception:
        raise Exception(f"Groq LLM invalid JSON: {r.text}")


def correct_grammar(text: str) -> str:
    """
    Correct grammar with priority:
    1. LanguageTool (offline, rule-based, free) ✓ RECOMMENDED
    2. HF FLAN-T5 transformer (offline, ML-based)
    3. Groq Llama (API, limited quota)
    """
    if not text or not isinstance(text, str):
        return text
    
    # Try LanguageTool first (best for this task - rule-based, offline, fast)
    if USE_LOCAL_LANGUAGE_TOOL:
        try:
            corrected = correct_with_language_tool(text)
            logger.info("Grammar correction: LanguageTool succeeded")
            return corrected
        except Exception as e:
            logger.warning(f"LanguageTool failed, trying alternatives: {e}")
    
    # Try HF transformer
    try:
        corrected = correct_with_hf_transformer(text)
        logger.info("Grammar correction: HF transformer succeeded")
        return corrected
    except Exception as e:
        logger.warning(f"HF transformer failed, trying Groq: {e}")
    
    # Fall back to Groq API
    try:
        corrected = correct_with_groq_llm(text)
        logger.info("Grammar correction: Groq API succeeded")
        return corrected
    except Exception as e:
        logger.error(f"All grammar correction methods failed: {e}")
        # Return original text if all methods fail
        logger.info("Returning original text (no correction applied)")
        return text
