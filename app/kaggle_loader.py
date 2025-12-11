import os
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "data")

def load_kaggle_samples():
    folder = os.path.join(BASE, "kaggle_samples")
    dfs = []

    for f in os.listdir(folder):
        if f.endswith(".csv"):
            dfs.append(pd.read_csv(os.path.join(folder, f)))

    return dfs

def load_submission_template():
    return pd.read_csv(os.path.join(BASE, "submission.csv"))
