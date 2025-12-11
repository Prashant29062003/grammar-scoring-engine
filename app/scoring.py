from jiwer import wer

def compute_wer_and_score(original: str, corrected: str):
    try:
        error = wer(original, corrected)
    except Exception:
        error = 1.0
    score = max(0.0, 1.0 - error) * 100.0
    return error, round(score, 2)


def batch_score(pairs: list[dict]):
    """
    pairs format:
    [
        {"original": "...", "corrected": "..."},
        ...
    ]
    """
    results = []
    for p in pairs:
        o = p["original"]
        c = p["corrected"]
        wer_value, score = compute_wer_and_score(o, c)
        results.append({
            "original": o,
            "corrected": c,
            "wer": round(wer_value, 4),
            "score": score
        })
    return results
