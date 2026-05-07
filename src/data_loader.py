import os
import pyreadr
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw')

DATASETS = [
    'vote_history', 'castaways', 'castaway_details',
    'challenge_results', 'advantage_movement',
    'confessionals', 'screen_time', 'tribe_mapping',
    'season_summary', 'jury_votes', 'boot_mapping', 'episodes',
]


def load_all(raw_dir=None):
    if raw_dir is None:
        raw_dir = RAW_DIR
    dfs = {}
    for name in DATASETS:
        path = os.path.join(raw_dir, f'{name}.rda')
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue
        result = pyreadr.read_r(path)
        df = list(result.values())[0]
        for col in df.select_dtypes(include='category').columns:
            df[col] = df[col].astype(str)
        dfs[name] = df
    print(f"Loaded {len(dfs)} datasets")
    return dfs
