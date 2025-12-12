# Grammar Scoring Engine - SHL Competition

## Quick Summary

This project trains a **supervised regression model** to predict grammar MOS scores (0-5) from 45-60 second audio files.

### Key Improvements Made
1. ✅ **Local Whisper** for transcription (offline, no quota limits, free)
2. ✅ **LanguageTool** for grammar correction (offline, rule-based, free)
3. ✅ **Enhanced error handling** with transcript caching
4. ✅ **Fill missing predictions** script to complete submission
5. ✅ **Required RMSE metric** prominently reported in notebook

---

## Installation & Setup

### 1. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# FOR BEST PERFORMANCE (Recommended):

# Local Whisper (no API keys, offline)
pip install openai-whisper

# Local LanguageTool (no API keys, offline)
pip install language-tool-python

# Optional: Groq fallback (if you have API access)
pip install groq

# Optional: Hugging Face transformers (fallback grammar correction)
pip install transformers torch
```

### 2. Configure `.env`

Copy `.env.sample` to `.env`:
```bash
cp .env.sample .env
```

Edit `.env` with your preferences:

**Best Setup (RECOMMENDED - Free, No Quotas):**
```env
USE_LOCAL_WHISPER=true
LOCAL_WHISPER_MODEL=base  # Options: tiny, base, small, medium, large
USE_LOCAL_LANGUAGE_TOOL=true

# Optional fallbacks (set to false if not using)
GROQ_API_KEY=
USE_HF_FALLBACK=false
HF_TOKEN=
```

**If You Have Groq API Access:**
```env
USE_LOCAL_WHISPER=false  # Disable local to use Groq
GROQ_API_KEY=your_api_key_here
GROQ_ASR_MODEL=whisper-large-v3
GROQ_LLM_MODEL=llama-3.3-70b-versatile
```

---

## Usage

### Option A: FastAPI Server

```bash
# Start server
uvicorn app.main:app --reload

# Then use Swagger UI at: http://127.0.0.1:8000/docs
# Endpoints:
#   POST /score/          - Score a single audio file
#   POST /train/evaluate  - Extract features from train data
#   POST /model/train     - Train the regression model
#   POST /model/predict-kaggle - Generate test predictions
```

### Option B: Jupyter Notebook

```bash
jupyter notebook notebooks/grammar_scoring_pipeline.ipynb
```

Run all cells to:
1. Load training features
2. Train RandomForest model
3. **Display RMSE on training data** (required metric)
4. Generate test predictions
5. Save `data/kaggle/submission.csv`

### Option C: Python Scripts

```bash
# Extract training features from audio
python -c "from app.train_evaluate import run_train_evaluation; print(run_train_evaluation())"

# Train model
python -c "from app.model_train import train_regression_model; print(train_regression_model())"

# Generate predictions
python -c "from app.model_predict import predict_kaggle_submission; print(predict_kaggle_submission())"

# Fill any missing predictions
python scripts/fill_missing_predictions.py
```

---

## Model Architecture

### Learning Type
**Supervised Regression** — Maps audio → continuous score [0, 5]

### Pipeline
```
Audio Files
    ↓
[Local Whisper / Groq API] → Transcript
    ↓
[LanguageTool / HF / Groq] → Corrected Text
    ↓
[Feature Extractor] → Fluency Features
    ├─ Word count
    ├─ Avg word length
    ├─ Filler count (um, uh, etc.)
    ├─ Repetitions
    └─ Punctuation
    ↓
[RandomForest Regressor] → Predicted Score [0, 5]
```

### Models Used

| Component | Primary | Fallback | Free | Offline |
|-----------|---------|----------|------|---------|
| **ASR** | Local Whisper | Groq Whisper | ✅ | ✅ |
| **Grammar** | LanguageTool | HF FLAN-T5 | ✅ | ✅ |
| **Regression** | RandomForest | — | ✅ | ✅ |

---

## Results & Metrics

### Required Metric
- **RMSE on Training Data**: Computed and reported in Jupyter notebook

### Additional Metrics
- RMSE on Validation Data
- MAE (Mean Absolute Error)
- R² Score
- Feature Importances

Run the notebook to see detailed results and visualizations.

---

## Submission Format

Final submission saved to: `data/kaggle/submission.csv`

Format (required):
```csv
filename,label
audio_1,3.45
audio_2,4.12
...
```

### Handling Missing Predictions

If any predictions are blank after running:

1. **Check debug log**: `data/kaggle/submission_debug.csv` for error reasons
2. **Run fill script**:
   ```bash
   python scripts/fill_missing_predictions.py
   ```
3. **Verify audio files exist**: `data/kaggle/test_audio/`

---

## Troubleshooting

### Issue: Blank predictions in submission.csv

**Cause**: ASR failed (missing GROQ_API_KEY, Whisper not installed, audio missing)

**Solution**:
```bash
# Install local Whisper
pip install openai-whisper

# Ensure .env has:
USE_LOCAL_WHISPER=true

# Fill missing predictions
python scripts/fill_missing_predictions.py
```

### Issue: "whisper not installed"

```bash
pip install openai-whisper torch
```

### Issue: "language-tool-python not installed"

```bash
pip install language-tool-python
```

### Issue: Groq API quota exceeded

**Solution**: Use local Whisper + LanguageTool instead:
```env
USE_LOCAL_WHISPER=true
USE_LOCAL_LANGUAGE_TOOL=true
GROQ_API_KEY=  # Leave blank
```

### Issue: Slow transcription

- Use smaller model: `LOCAL_WHISPER_MODEL=tiny` (faster, less accurate)
- Or use `base` (good balance)
- Install GPU support: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

---

## Project Structure

```
.
├── .env.sample              # Configuration template (UPDATED)
├── requirements.txt         # Dependencies
├── app/
│   ├── config.py           # Configuration (UPDATED)
│   ├── transcriber.py      # Original Groq-only transcriber
│   ├── transcriber_enhanced.py  # NEW: Local Whisper + caching
│   ├── grammar.py          # Original Groq-only grammar corrector
│   ├── grammar_enhanced.py  # NEW: LanguageTool + HF + Groq
│   ├── train_evaluate.py   # Feature extraction
│   ├── model_train.py      # Train regression model
│   ├── model_predict.py    # Generate predictions (UPDATED)
│   └── main.py             # FastAPI server
├── notebooks/
│   └── grammar_scoring_pipeline.ipynb  # Main notebook (UPDATED)
├── scripts/
│   └── fill_missing_predictions.py     # NEW: Fill blanks in submission
├── data/
│   ├── model.pkl           # Trained model
│   ├── transcripts_cache/  # Cached transcripts (new)
│   └── kaggle/
│       ├── train.csv
│       ├── test.csv
│       ├── train_features.csv
│       └── submission.csv
└── README.md               # This file
```

---

## Model Comparison & Why Local Whisper

| Model | Speed | Accuracy | Cost | Offline | Quota Limits |
|-------|-------|----------|------|---------|--------------|
| **Local Whisper** | ~2-5s/file | 95% | Free | ✅ | None |
| **Groq Whisper** | ~1s/file | 96% | Free tier | ❌ | 25 req/min |
| **OpenAI Whisper API** | ~1s/file | 96% | $0.36/hr | ❌ | None |

**Recommendation**: Use **Local Whisper** for reliability and cost.

---

## Next Steps / Future Improvements

1. **Better feature engineering**:
   - Add POS-tag ratios, parse tree complexity
   - Use sentence-transformer embeddings for semantic features
   - Grammar-error counts via LanguageTool or LLM

2. **Advanced models**:
   - Transfer learning: Fine-tune Wav2Vec2 / HuBERT on grammar scores
   - Ensemble: Combine RandomForest + XGBoost + Neural Net

3. **Data augmentation**:
   - Synthetic audio with noise/speed variations
   - Back-translation for text augmentation

4. **Evaluation**:
   - Stratified K-fold cross-validation
   - Spearman rank correlation (for ordinal consistency)
   - Error analysis by audio quality / speaker profile

---

## Contact & Support

For issues or questions:
1. Check `data/kaggle/submission_debug.csv` for detailed error logs
2. Review `.env` configuration
3. Ensure all dependencies are installed: `pip list | grep -E "whisper|language|sklearn|pandas"`

---

**Last Updated**: December 2025
