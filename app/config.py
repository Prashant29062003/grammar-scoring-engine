import os
from dotenv import load_dotenv
from pathlib import Path

# load .env from project root
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ==================== LOCAL/OFFLINE OPTIONS ====================
# Use local Whisper (offline, no API quota limits) - RECOMMENDED
USE_LOCAL_WHISPER = os.getenv("USE_LOCAL_WHISPER", "true").lower() in ("1", "true", "yes")
LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "base").strip()  # tiny, base, small, medium, large

# Use local LanguageTool (offline, rule-based grammar) - RECOMMENDED
USE_LOCAL_LANGUAGE_TOOL = os.getenv("USE_LOCAL_LANGUAGE_TOOL", "true").lower() in ("1", "true", "yes")

# ==================== FALLBACK APIS ====================
# GROQ (limited tier ~25 req/min free)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_ASR_MODEL = os.getenv("GROQ_ASR_MODEL", "whisper-large-v3").strip()
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile").strip()

# Hugging Face fallback
USE_HF_FALLBACK = os.getenv("USE_HF_FALLBACK", "true").lower() in ("1", "true", "yes")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_GRAMMAR_MODEL = os.getenv("HF_GRAMMAR_MODEL", "pszemraj/flan-t5-base-grammar-synthesis").strip()

# ==================== PERFORMANCE SETTINGS ====================
MAX_CHARS = int(os.getenv("MAX_CHARS", "500"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))

# ==================== LOGGING ====================
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
