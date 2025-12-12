from app.grammar_enhanced import correct_grammar
from app.scoring import compute_wer_and_score

def score_text_item(text: str):
    try:
        corrected = correct_grammar(text)
        wer_value, score = compute_wer_and_score(text, corrected)
        return {
            "input": text,
            "corrected": corrected,
            "wer": round(wer_value, 4),
            "score": score
        }
    except Exception as e:
        return {
            "input": text,
            "error": str(e)
        }

def score_text_batch(texts: list[str]):
    results = []
    for t in texts:
        results.append(score_text_item(t))
    return results
