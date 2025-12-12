import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.kaggle_loader import load_train_audio_path
from app.transcriber_enhanced import transcribe_from_path, transcribe_batch
from app.config import BATCH_SIZE

# ---- Feature extractor ----
def extract_fluency_features(text: str):
    words = text.split()
    len_words = len(words)

    fillers = ["uh", "um", "erm", "hmm"]
    filler_count = sum(1 for w in words if w.lower() in fillers)

    repetitions = sum(1 for i in range(1, len_words) if words[i] == words[i-1])

    avg_word_len = sum(len(w) for w in words) / len_words if len_words else 0

    punctuation_count = sum(text.count(p) for p in [".", ",", "?", "!"])

    return {
        "len_words": len_words,
        "avg_word_len": avg_word_len,
        "fillers": filler_count,
        "repetitions": repetitions,
        "punctuation": punctuation_count,
    }


# ----- Single file evaluation -----
def process_single_train(row):
    filename, true_label = row["filename"], row["label"]

    audio_path = load_train_audio_path(filename)
    if audio_path is None:
        return {
            "filename": filename,
            "true_label": true_label,
            "error": "file_not_found"
        }

    try:
        asr_text = transcribe_from_path(audio_path)
        feats = extract_fluency_features(asr_text)

        return {
            "filename": filename,
            "true_label": true_label,
            "asr_text": asr_text,
            **feats
        }

    except Exception as e:
        return {
            "filename": filename,
            "true_label": true_label,
            "error": str(e)
        }


# ----- Main threaded evaluation -----
def run_train_evaluation():
    df = pd.read_csv("data/kaggle/train.csv")

    # Prepare list of audio paths for batch transcription
    audio_paths = []
    filename_to_path = {}
    for _, row in df.iterrows():
        fn = row["filename"]
        p = load_train_audio_path(fn)
        filename_to_path[fn] = p
        if p is not None:
            audio_paths.append(p)

    # Transcribe in parallel (uses local whisper worker pool)
    transcripts = {}
    if audio_paths:
        try:
            transcripts = transcribe_batch(audio_paths, max_workers=BATCH_SIZE)
        except Exception:
            # Fallback: try single-threaded transcriptions
            transcripts = {}
            for p in audio_paths:
                try:
                    transcripts[p] = transcribe_from_path(p)
                except Exception as e:
                    transcripts[p] = None

    results = []
    for _, row in df.iterrows():
        filename, true_label = row["filename"], row["label"]
        audio_path = filename_to_path.get(filename)
        if audio_path is None:
            results.append({
                "filename": filename,
                "true_label": true_label,
                "error": "file_not_found"
            })
            continue

        asr_text = transcripts.get(audio_path)
        if not asr_text:
            results.append({
                "filename": filename,
                "true_label": true_label,
                "error": "transcription_failed"
            })
            continue

        try:
            feats = extract_fluency_features(asr_text)
            results.append({
                "filename": filename,
                "true_label": true_label,
                "asr_text": asr_text,
                **feats
            })
        except Exception as e:
            results.append({
                "filename": filename,
                "true_label": true_label,
                "error": str(e)
            })

    out_path = "data/kaggle/train_features.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    return out_path
