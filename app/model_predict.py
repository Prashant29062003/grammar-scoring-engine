import os
import pandas as pd
import joblib
import logging
from app.transcriber_enhanced import transcribe_from_path
from app.train_evaluate import extract_fluency_features
from app.kaggle_loader import load_test_audio_path

MODEL_PATH = "data/model.pkl"
TEST_CSV = "data/kaggle/test.csv"
OUTPUT_SUBMISSION = "data/kaggle/submission.csv"
DEBUG_SUBMISSION = "data/kaggle/submission_debug.csv"


def predict_kaggle_submission():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Train model first using /model/train")

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(TEST_CSV)
    results = []
    debug_rows = []

    # basic logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    for _, row in df.iterrows():
        filename = row["filename"]
        audio_path = load_test_audio_path(filename)

        if audio_path is None:
            results.append({"filename": filename, "label": ""})
            debug_rows.append({"filename": filename, "label": "", "error": "audio_file_missing"})
            logger.warning("Missing audio file for %s", filename)
            continue

        try:
            asr = transcribe_from_path(audio_path)
            feats = extract_fluency_features(asr)

            features = [[
                feats["len_words"],
                feats["avg_word_len"],
                feats["fillers"],
                feats["repetitions"],
                feats["punctuation"],
            ]]

            pred = float(model.predict(features)[0])
            # Clip prediction to valid grammar score range [0, 5]
            pred = min(5.0, max(0.0, pred))

            results.append({
                "filename": filename,
                "label": round(pred, 3)
            })
            debug_rows.append({"filename": filename, "label": round(pred, 3), "error": ""})

        except Exception as e:
            # record the error for debugging; keep official submission format unchanged
            results.append({"filename": filename, "label": ""})
            debug_rows.append({"filename": filename, "label": "", "error": str(e)})
            logger.exception("Prediction failed for %s: %s", filename, e)

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_SUBMISSION, index=False)

    # write debug CSV with error messages to help diagnose missing predictions
    try:
        dbg = pd.DataFrame(debug_rows)
        dbg.to_csv(DEBUG_SUBMISSION, index=False)
        logger.info("Wrote debug submission to %s", DEBUG_SUBMISSION)
    except Exception:
        logger.exception("Failed writing debug submission file")

    # summary
    missing = sum(1 for r in results if r.get('label') in (None, ""))
    logger.info("Total test rows: %d, missing labels: %d", len(results), missing)
    logger.info("Submission saved to %s", OUTPUT_SUBMISSION)

    return OUTPUT_SUBMISSION
