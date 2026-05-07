"""
Weekly prediction script for the current Survivor season.

Usage:
    python -m src.predict [--season 50] [--episode N]

Outputs normalized win probabilities for all active players,
plus week-over-week delta if a previous episode's prediction exists.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb

from src.features import FEATURE_COLS, build_prediction_snapshots
from src.data_loader import load_all

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')


def load_model():
    path = os.path.join(MODELS_DIR, 'survivor_model.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run training first.")
    model = xgb.XGBClassifier()
    model.load_model(path)
    return model


def normalize_probs(df, prob_col='raw_prob'):
    total = df[prob_col].sum()
    df = df.copy()
    df['win_probability'] = df[prob_col] / max(total, 1e-9)
    return df


def predict_current_season(dfs=None, season=50, verbose=True):
    if dfs is None:
        dfs = load_all()

    model = load_model()

    active = build_prediction_snapshots(dfs, current_season=season)
    if len(active) == 0:
        print(f"No active players found for season {season}.")
        return None

    X = active[FEATURE_COLS].fillna(0).values
    raw_probs = model.predict_proba(X)[:, 1]

    results = active[['season', 'episode', 'castaway_id', 'castaway']].copy()
    results['raw_prob'] = raw_probs
    results = normalize_probs(results)
    results = results.sort_values('win_probability', ascending=False).reset_index(drop=True)
    results['rank'] = results.index + 1

    # Key feature drivers for interpretability
    feature_importance_path = os.path.join(MODELS_DIR, 'feature_importance.json')
    if os.path.exists(feature_importance_path):
        with open(feature_importance_path) as f:
            importance = json.load(f)
        top_features = list(importance.keys())[:5]
    else:
        top_features = FEATURE_COLS[:5]

    for feat in top_features:
        if feat in active.columns:
            results[feat] = active[feat].values

    # Week-over-week delta
    ep = int(results['episode'].iloc[0])
    prev_path = os.path.join(PROCESSED_DIR, f'predictions_s{season}_ep{ep - 1}.csv')
    if os.path.exists(prev_path) and ep > 1:
        prev = pd.read_csv(prev_path)
        prev_map = prev.set_index('castaway_id')['win_probability'].to_dict()
        results['prev_prob'] = results['castaway_id'].map(prev_map)
        results['prob_delta'] = results['win_probability'] - results['prev_prob']
    else:
        results['prob_delta'] = float('nan')

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, f'predictions_s{season}_ep{ep}.csv')
    results.to_csv(out_path, index=False)

    if verbose:
        print(f"\nSeason {season} — Episode {ep} Win Probabilities")
        print("=" * 60)
        for _, row in results.iterrows():
            delta_str = ''
            if not pd.isna(row.get('prob_delta', float('nan'))):
                d = row['prob_delta']
                delta_str = f"  ({'▲' if d >= 0 else '▼'}{abs(d):.1%})"
            print(f"  {row['rank']:>2}. {row['castaway']:<20} {row['win_probability']:.1%}{delta_str}")
        print(f"\nSaved → {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Survivor win probability predictions')
    parser.add_argument('--season', type=int, default=50)
    parser.add_argument('--episode', type=int, default=None, help='Ignored; uses latest available data')
    args = parser.parse_args()

    dfs = load_all()
    predict_current_season(dfs, season=args.season)


if __name__ == '__main__':
    main()
