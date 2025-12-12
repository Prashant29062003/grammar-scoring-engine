import os
from typing import List

# Default paths for your batch audio processing
DEFAULT_BATCH_AUDIO_DIR = "data/kaggle_samples/audio"

# For SHL Kaggle competition
KAGGLE_TRAIN_AUDIO_DIR = "data/kaggle/train_audio"
KAGGLE_TEST_AUDIO_DIR = "data/kaggle/test_audio"

TRAIN_AUDIO_DIR = "data/kaggle/train_audio"
TEST_AUDIO_DIR = "data/kaggle/test_audio"

def load_train_audio_path(filename: str):
    """Return full path to train audio file, or None if missing"""
    path = os.path.join(TRAIN_AUDIO_DIR, filename + ".wav")
    if os.path.exists(path):
        return path
    return None

def load_test_audio_path(filename: str):
    """Return full path to test audio file, or None if missing"""
    path = os.path.join(TEST_AUDIO_DIR, filename + ".wav")
    if os.path.exists(path):
        return path
    return None

def load_audio_files(directory: str = DEFAULT_BATCH_AUDIO_DIR) -> List[str]:
    """
    Loads all audio file paths from a given directory.
    Default: data/kaggle_samples/audio
    """
    if not os.path.exists(directory):
        return []

    valid_ext = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

    audio_files = []
    for f in os.listdir(directory):
        if f.lower().endswith(valid_ext):
            audio_files.append(os.path.join(directory, f))

    return sorted(audio_files)


def load_train_audio_files() -> List[str]:
    """
    Loads all audio files from data/kaggle/train_audio
    """
    return load_audio_files(KAGGLE_TRAIN_AUDIO_DIR)


def load_test_audio_files() -> List[str]:
    """
    Loads all audio files from data/kaggle/test_audio
    """
    return load_audio_files(KAGGLE_TEST_AUDIO_DIR)
