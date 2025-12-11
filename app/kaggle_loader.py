import csv
import os

BASE = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.normpath(os.path.join(BASE, 'kaggle_data'))

# dataset.csv columns: bad_sentence,corrected_sentence

def load_dataset_pairs():
    path = os.path.join(DATA_DIR, 'dataset.csv')
    if not os.path.exists(path):
        raise FileNotFoundError('dataset.csv not found in kaggle_data')
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i,row in enumerate(reader):
            audio_name = f"{str(i+1).zfill(2)}_bad.wav"
            audio_path = os.path.join(DATA_DIR, 'audio', audio_name)
            rows.append({'audio_path': audio_path, 'bad': row.get('bad_sentence',''), 'corrected': row.get('corrected_sentence','')})
    return rows