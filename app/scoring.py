from jiwer import wer

def compute_wer_and_score(original: str, corrected: str):
    """
    Compute WER and a grammar score 0-100.
    Score = max(0, 1 - wer) * 100
    """
    try:
        error = wer(original, corrected)
    except Exception:
        # fallback: treat as maximum error
        error = 1.0
    score = max(0.0, 1.0 - error) * 100.0
    return error, round(score, 2)
