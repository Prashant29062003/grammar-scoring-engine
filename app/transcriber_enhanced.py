"""
Enhanced transcriber with local Whisper fallback and caching.
Priority: Local Whisper (offline, no quotas) → Groq Whisper (API, limited) → Error
"""
import os
import json
import logging
from pathlib import Path
from app.config import (
    GROQ_API_KEY, GROQ_ASR_MODEL, REQUEST_TIMEOUT,
    USE_LOCAL_WHISPER, LOCAL_WHISPER_MODEL
)

logger = logging.getLogger(__name__)

# Transcript cache directory
CACHE_DIR = Path("data/transcripts_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ==================== LOCAL WHISPER ====================
def transcribe_with_local_whisper(audio_path: str) -> str:
    """Transcribe using local OpenAI Whisper (offline, no API quota limits)."""
    try:
        import whisper
    except ImportError:
        logger.error("whisper not installed. Run: pip install openai-whisper")
        raise ImportError("Install openai-whisper: pip install openai-whisper")
    
    try:
        # Load model once (cached after first load)
        model = whisper.load_model(LOCAL_WHISPER_MODEL)
        result = model.transcribe(audio_path, language="en", verbose=False)
        text = result["text"].strip()
        logger.info(f"Local Whisper transcribed {audio_path}: {len(text)} chars")
        return text
    except Exception as e:
        logger.error(f"Local Whisper failed for {audio_path}: {e}")
        raise


def transcribe_with_groq_api(audio_bytes: bytes) -> str:
    """Transcribe using Groq API (fallback, limited quota)."""
    import requests
    
    if not GROQ_API_KEY:
        raise Exception("Missing GROQ_API_KEY in environment")

    GROQ_ASR_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
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

    text = payload.get("text") or payload.get("transcription") or payload.get("result")
    return text or str(payload)


def get_cache_path(audio_path: str) -> Path:
    """Get cache file path for audio transcript."""
    filename = Path(audio_path).stem + ".json"
    return CACHE_DIR / filename


def load_from_cache(audio_path: str) -> str:
    """Load transcript from cache if available."""
    cache_path = get_cache_path(audio_path)
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
                logger.info(f"Loaded cached transcript for {audio_path}")
                return data.get("text", "")
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")
    return None


def save_to_cache(audio_path: str, text: str):
    """Save transcript to cache."""
    cache_path = get_cache_path(audio_path)
    try:
        with open(cache_path, "w") as f:
            json.dump({"text": text, "audio": str(audio_path)}, f, indent=2)
        logger.info(f"Cached transcript for {audio_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_path}: {e}")


def transcribe_from_path(audio_path: str) -> str:
    """
    Transcribe audio with priority:
    1. Check cache
    2. Use local Whisper (if enabled)
    3. Fall back to Groq API (if available)
    """
    # Try cache first
    cached = load_from_cache(audio_path)
    if cached:
        return cached
    
    text = None
    
    # Try local Whisper first (no quota limits, offline)
    if USE_LOCAL_WHISPER:
        try:
            text = transcribe_with_local_whisper(audio_path)
            save_to_cache(audio_path, text)
            return text
        except Exception as e:
            logger.warning(f"Local Whisper failed, trying Groq: {e}")
    
    # Fall back to Groq API
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        text = transcribe_with_groq_api(audio_bytes)
        save_to_cache(audio_path, text)
        return text
    except Exception as e:
        logger.error(f"All transcription methods failed for {audio_path}: {e}")
        raise


def transcribe_bytes_from_bytes(audio_bytes: bytes) -> str:
    """Transcribe from raw bytes (used by FastAPI endpoints).

    Writes bytes to a temporary WAV file and calls transcribe_from_path so
    local Whisper can be used when enabled (and caching works).
    """
    import tempfile
    import os
    # On Windows NamedTemporaryFile keeps the file open which can cause
    # permission errors when another reader/process tries to open it.
    # Use mkstemp + close the fd, then remove file in finally block.
    suffix = ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        # write bytes using the file descriptor then close it
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
            f.flush()
        return transcribe_from_path(path)
    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.warning(f"Failed to remove temp file {path}: {e}")


# --------------------------
# Parallel batch transcription
# --------------------------
def _transcribe_file_worker(args):
    """Worker function for ProcessPoolExecutor: loads model and transcribes a single file.
    args: tuple(audio_path, model_name)
    """
    audio_path, model_name = args
    try:
        import whisper
        model = whisper.load_model(model_name)
        res = model.transcribe(audio_path, language='en', verbose=False)
        text = res.get('text', '').strip()
        return (audio_path, text, None)
    except Exception as e:
        return (audio_path, None, str(e))


def transcribe_batch(audio_paths, max_workers=2, model_name=None):
    """Transcribe a list of audio file paths in parallel using multiple processes.

    - Checks cache first and only transcribes missing entries.
    - Uses ProcessPoolExecutor to avoid GIL limitations.
    - Returns dict: {audio_path: transcript}
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if model_name is None:
        model_name = LOCAL_WHISPER_MODEL

    # Prepare results dict, load cached where available
    results = {}
    to_process = []
    for p in audio_paths:
        cached = load_from_cache(p)
        if cached:
            results[p] = cached
        else:
            to_process.append(p)

    if not to_process:
        logger.info("All %d transcripts loaded from cache", len(audio_paths))
        return results

    logger.info("Transcribing %d files in parallel (workers=%d)", len(to_process), max_workers)

    # Prepare worker args
    worker_args = [(p, model_name) for p in to_process]

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        future_to_path = {exe.submit(_transcribe_file_worker, arg): arg[0] for arg in worker_args}
        for fut in as_completed(future_to_path):
            path = future_to_path[fut]
            try:
                p, text, err = fut.result()
                if err:
                    logger.error("Transcription failed for %s: %s", p, err)
                else:
                    results[p] = text
                    save_to_cache(p, text)
                    logger.info("Transcribed and cached %s", p)
            except Exception as e:
                logger.exception("Worker failed for %s: %s", path, e)

    return results
