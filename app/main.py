from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import sys, os
import numpy as _np
from app.transcriber import transcribe_bytes_from_bytes, transcribe_from_path
from app.grammar import correct_grammar
from app.scoring import compute_wer_and_score, batch_score
from app.kaggle_loader import load_dataset_pairs
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

@app.post('/score/')
async def score_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
        raise HTTPException(status_code=400, detail='Upload a valid audio file')

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail='Empty file uploaded')

    try:
        asr_text = transcribe_bytes_from_bytes(audio_bytes)
        corrected_text = correct_grammar(asr_text)
        wer_val, score = compute_wer_and_score(asr_text, corrected_text)
        return JSONResponse({
            'filename': file.filename,
            'asr_text': asr_text,
            'corrected_text': corrected_text,
            'wer': round(wer_val, 4),
            'grammar_score_0_100': score
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Processing failed: {e}')

# Batch: process local kaggle_data/audio using dataset.csv
@app.post('/batch/process-kaggle')
def batch_process_kaggle():
    pairs = load_dataset_pairs()
    results = []
    for p in pairs:
        audio_path = p['audio_path']
        if not os.path.exists(audio_path):
            results.append({**p, 'error': 'audio missing'})
            continue
        try:
            asr = transcribe_from_path(audio_path)
            corrected = correct_grammar(asr)
            wer_val, score = compute_wer_and_score(asr, corrected)
            results.append({
                'audio': os.path.basename(audio_path),
                'dataset_bad': p['bad'],
                'dataset_corrected': p['corrected'],
                'asr_text': asr,
                'corrected_text': corrected,
                'wer': round(wer_val, 4),
                'score': score
            })
        except Exception as e:
            results.append({**p, 'error': str(e)})
    out_path = os.path.join('data', 'kaggle_samples', 'submission_results.csv')
    save_results_csv(results, out_path)
    return {'message': 'done', 'processed': len(results), 'csv': out_path}

@app.get('/batch/download-submission')
def download_submission():
    p = 'data/submission_results.csv'
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail='submission not found')
    return FileResponse(p, media_type='text/csv', filename='submission_results.csv')