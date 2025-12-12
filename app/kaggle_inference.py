import os
from app.transcriber_enhanced import transcribe_from_path
from app.grammar_enhanced import correct_grammar

TEST_AUDIO_DIR = "data/kaggle/test_audio"
TEST_CSV = "data/kaggle/test.csv"
OUTPUT_CSV = "data/kaggle/submission.csv"

def run_kaggle_inference():
    df = pd.read_csv(TEST_CSV)

    predictions = []

    for _, row in df.iterrows():
        filename = row["filename"]
        audio_path = os.path.join(TEST_AUDIO_DIR, filename)

        if not os.path.exists(audio_path):
            predictions.append({"filename": filename, "prediction": ""})
            continue

        try:
            asr = transcribe_from_path(audio_path)
            corrected = correct_grammar(asr)
        except Exception:
            corrected = ""

        predictions.append({
            "filename": filename,
            "prediction": corrected
        })

    out_df = pd.DataFrame(predictions)
    out_df.to_csv(OUTPUT_CSV, index=False)

    return OUTPUT_CSV
