import os
from dotenv import load_dotenv
from pathlib import Path

# load .env from project root
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# GROQ (required)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_ASR_MODEL = os.getenv("GROQ_ASR_MODEL", "whisper-large-v3").strip()
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "gpt-4o-mini").strip()  # change if you have specific

# Optional: HF fallback (only if you want HF fallback)
USE_HF_FALLBACK = os.getenv("USE_HF_FALLBACK", "false").lower() in ("1","true","yes")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
HF_GRAMMAR_MODEL = os.getenv("HF_GRAMMAR_MODEL", "pszemraj/flan-t5-base-grammar-synthesis").strip()

# app behavior
MAX_CHARS = int(os.getenv("MAX_CHARS", "500"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))  # seconds
