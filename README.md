# Grammar Scoring Engine — SHL

This repository provides a pipeline to predict grammar Mean Opinion Scores (MOS) from short audio files. It converts audio → transcript → corrected text → engineered features → regression model → predicted score (0–5). The project supports local (OpenAI Whisper) and API (Groq) transcription, LanguageTool-based grammar correction, caching, and batch Kaggle-style processing.

**Highlights**
- Local Whisper support (offline, no quota)
- LanguageTool grammar checks and HF fallback
- Transcript caching to speed repeated runs
- Scripts and FastAPI endpoints for scoring, training, and batch inference

**Quick Links**
- File: [app/transcriber_enhanced.py](app/transcriber_enhanced.py)
- File: [app/main.py](app/main.py)
- Folder: [data/kaggle](data/kaggle)

**Installation**
1. Create and activate virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Unix
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

Optional recommended packages:

```bash
pip install openai-whisper language-tool-python
```

**Configuration**
Copy `.env.sample` to `.env` and adjust settings. Important variables:

- `USE_LOCAL_WHISPER` — set `true` to use local Whisper model
- `LOCAL_WHISPER_MODEL` — `tiny|base|small|medium|large`
- `GROQ_API_KEY` — Groq fallback API key (optional)

**Running Locally**

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Open the Swagger UI at `http://127.0.0.1:8000/docs`.

Key endpoints:

- `POST /score/` — score a single audio file (multipart/form-data `file`)
- `POST /model/train` — train the regression model
- `POST /model/predict-kaggle` — generate Kaggle-style predictions

**Scripts / Notebooks**

- `notebooks/grammar_scoring_pipeline.ipynb` — end-to-end pipeline and evaluation
- `scripts/fill_missing_predictions.py` — fills blanks in submission.csv

**Data & Output**

- Input: `data/kaggle/test_audio/` and `data/kaggle/train_audio/`
- Model artifacts: `data/model.pkl`
- Submissions: `data/kaggle/submission.csv`, debug: `data/kaggle/submission_debug.csv`

**Troubleshooting**

- Permission errors writing temporary WAV files on Windows: see updated `app/transcriber_enhanced.py` (uses `tempfile.mkstemp`).
- If `whisper` is missing: `pip install openai-whisper`
- If `scikit-learn` issues occur, ensure `requirements.txt` uses `scikit-learn` (not `sklearn`).

**Development Notes**

- Caching: transcripts are stored in `data/transcripts_cache/` to avoid repeated ASR calls.
- Use smaller Whisper models for speed in development (`tiny` or `base`).

**Next Steps / Improvements**

- Add richer linguistic features (POS ratios, parse complexity).
- Experiment with transformer embeddings and ensemble regressors.

---

For details on usage and advanced options, see `README_ENHANCED.md`.