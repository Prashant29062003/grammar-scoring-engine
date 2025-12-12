import os

AUDIO_DIR = os.path.join("data", "kaggle_samples", "audio")

def load_audio_files():
    if not os.path.exists(AUDIO_DIR):
        return []

    return [
        os.path.join(AUDIO_DIR, f)
        for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg"))
    ]
