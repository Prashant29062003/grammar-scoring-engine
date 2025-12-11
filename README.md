# Speech Evaluation API (Groq ASR + Groq LLM Grammar + WER)

This FastAPI project transcribes audio using Groq Whisper, corrects grammar using Groq LLM (with optional Hugging Face fallback), computes WER and returns a grammar score.

## Quick start (local)

1. Clone repository.
2. Create virtualenv:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
````

3. Install:

   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` → `.env` and fill:

   * `GROQ_API_KEY` (required)
   * (optional) `USE_HF_FALLBACK=true` and `HF_TOKEN` if you want HF fallback
5. Start server:

   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
6. Visit Swagger UI: `http://127.0.0.1:8000/docs` — upload `sample.wav` and test.

## Required tokens & APIs

### Groq

* `GROQ_API_KEY` — get from Groq dashboard.
* Use `whisper-large-v3` for ASR.
* Use `gpt-4o-mini` or other Groq LLM for grammar.

### (Optional) Hugging Face Router

* `HF_TOKEN` — only if `USE_HF_FALLBACK=true`.
* Ensure token has **Make calls to Inference Providers** permission.
* Use `HF_GRAMMAR_MODEL` that is router-enabled; otherwise fallback may fail.

## Avoiding 404/500 errors

* Use **Groq** for both ASR and grammar to avoid HF router 404 issues.
* Ensure tokens are set in `.env` and contain correct permissions.
* Increase `REQUEST_TIMEOUT` if your network is slow.
* Check `/debug` endpoint to verify tokens are read.

## Deploying (suggested free hosts)

* Render, Railway, or Fly.io. Use Dockerfile for container deploy.
* Make sure to add environment variables in host's dashboard.

## Testing

Use `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/score/" -F "file=@./sample.wav"
```

