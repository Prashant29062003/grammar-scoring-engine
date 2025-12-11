# Grammar Scoring Engine

```This repo contains a scoring engine: audio -> ASR (Groq) -> grammar correction (Groq LLM with HF fallback) -> WER & grammar score. Includes batch Kaggle processing.```

## Quick start (local)
1. Clone repo

```
git clone <repo>
cd grammar-scoring-engine
```
2. Create virtualenv
```
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill keys

4. Add your 30 audio files and dataset.csv under `kaggle_data/`:
```
kaggle_data/audio/01_bad.wav ... 30_bad.wav
kaggle_data/dataset.csv
```
5. Run app
```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
6. Open `http://127.0.0.1:8000/docs` to test endpoints.

## Batch processing (local)
Call POST `/batch/process-kaggle` (will write `kaggle_data/submission_results.csv`).