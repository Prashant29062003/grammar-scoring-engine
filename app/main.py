from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import sys
import numpy as _np
from pydantic import BaseModel
from app.scoring import batch_score
from app.kaggle_loader import load_kaggle_samples, load_submission_template

from app.transcriber import transcribe_bytes
from app.grammar import correct_grammar
from app.scoring import compute_wer_and_score
from app.config import GROQ_API_KEY, HF_TOKEN


class BatchRequest(BaseModel):
    pairs: list[dict]   # [{ "original": "...", "corrected": "..." }]


app = FastAPI(
    title="Speech Evaluation API",
    description="ASR (Groq) → Grammar Correction (Groq LLM with optional HF fallback) → WER & Score",
    version="1.0.0"
)

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/debug")
def debug():
    return {
        "python": sys.executable,
        "numpy_available": True,
        "numpy_version": _np.__version__,
        "groq_configured": bool(GROQ_API_KEY),
        "hf_configured": bool(HF_TOKEN)
    }

@app.post("/score/")
async def score_endpoint(file: UploadFile = File(...)):
    # validate extension
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
        raise HTTPException(status_code=400, detail="Upload a valid audio file (.wav/.mp3/.m4a/.flac/.ogg)")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        asr_text = transcribe_bytes(audio_bytes)
        corrected_text = correct_grammar(asr_text)
        wer_value, score = compute_wer_and_score(asr_text, corrected_text)
        return JSONResponse({
            "filename": file.filename,
            "asr_text": asr_text,
            "corrected_text": corrected_text,
            "wer": round(wer_value, 4),
            "grammar_score_0_100": score
        })
    except Exception as e:
        # expose message for debugging (safe for submission) — if you want hide details, remove str(e)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/score/batch")
def score_batch_endpoint(body: BatchRequest):
    """
    Batch scoring for Kaggle-like text pairs.
    """
    return {"results": batch_score(body.pairs)}


@app.get("/kaggle/samples")
def kaggle_samples():
    dfs = load_kaggle_samples()
    return {
        "files_loaded": len(dfs),
        "rows_per_file": [len(df) for df in dfs]
    }


@app.get("/kaggle/submission-template")
def template():
    df = load_submission_template()
    return {"columns": df.columns.tolist()}