from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import sys, os
import numpy as _np
from app.transcriber import transcribe_bytes_from_bytes, transcribe_from_path
from app.grammar import correct_grammar
from app.scoring import compute_wer_and_score, batch_score
from app.kaggle_loader import load_audio_files
from app.utils import save_results_csv

app = FastAPI(title="Grammar Scoring Engine",
              description="ASR (Groq) → Grammar (Groq LLM/HF fallback) → WER & Score",
              version="1.0.0")

@app.get('/health')
def health():
    return {"status": "ok"}

@app.get('/debug')
def debug():
    return {
        "python": sys.executable,
        "numpy_version": _np.__version__,
    }

# -----------------------------
# Single Audio Scoring API
# -----------------------------
@app.post("/score/")
async def score_endpoint(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
        raise HTTPException(status_code=400, detail="Upload a valid audio file")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        asr_text = transcribe_bytes_from_bytes(audio_bytes)
        corrected_text = correct_grammar(asr_text)
        wer_val, score = compute_wer_and_score(asr_text, corrected_text)

        return JSONResponse({
            "filename": file.filename,
            "asr_text": asr_text,
            "corrected_text": corrected_text,
            "wer": round(wer_val, 4),
            "grammar_score_0_100": score
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Batch Processing: All files in data/kaggle_samples/audio
# -----------------------------
@app.post("/batch/process-audio")
def batch_process_audio():

    audio_files = load_audio_files()

    if not audio_files:
        return {"message": "done", "processed": 0, "note": "no audio files found"}

    results = []

    for audio_path in audio_files:
        try:
            asr = transcribe_from_path(audio_path)
            corrected = correct_grammar(asr)
            wer_val, score = compute_wer_and_score(asr, corrected)

            results.append({
                "audio": os.path.basename(audio_path),
                "asr_text": asr,
                "corrected_text": corrected,
                "wer": round(wer_val, 4),
                "score": score
            })

        except Exception as e:
            results.append({
                "audio": os.path.basename(audio_path),
                "error": str(e)
            })

    out_path = os.path.join("data", "submission_results.csv")
    save_results_csv(results, out_path)

    return {
        "message": "done",
        "processed": len(results),
        "csv": out_path
    }


@app.get("/batch/download")
def download_csv():
    p = os.path.join("data", "submission_results.csv")

    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="results not found")

    return FileResponse(
        p,
        media_type="text/csv",
        filename="submission_results.csv"
    )