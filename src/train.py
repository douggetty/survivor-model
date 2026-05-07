"""
XGBoost training with leave-one-season-out cross-validation.

Saves the full-data model to models/survivor_model.json and
LOSO evaluation results to data/processed/loso_results.csv.
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss

from src.features import FEATURE_COLS

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')


def _make_xgb(scale_pos_weight=10, seed=42):
    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=30,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )


def _normalize_probs(df, prob_col='raw_prob'):
    """Normalize raw model scores to sum to 1 within each (season, episode)."""
    df = df.copy()
    totals = df.groupby(['season', 'episode'])[prob_col].transform('sum')
    df['win_probability'] = df[prob_col] / totals.clip(lower=1e-9)
    return df


def loso_cross_validation(feature_df):
    """
    Leave-one-season-out cross-validation.

    Returns a DataFrame with per-season metrics and a full predictions DataFrame.
    """
    seasons = sorted(feature_df['season'].unique())
    print(f"Running LOSO CV across {len(seasons)} seasons …")

    # Global scale_pos_weight
    n_pos = feature_df['won_season'].sum()
    n_neg = len(feature_df) - n_pos
    spw = n_neg / max(n_pos, 1)

    all_preds = []
    season_metrics = []

    for held_out in seasons:
        train_df = feature_df[feature_df['season'] != held_out]
        test_df = feature_df[feature_df['season'] == held_out]

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df['won_season'].values
        X_test = test_df[FEATURE_COLS].values
        y_test = test_df['won_season'].values

        # Use last 15% of training rows (ordered by season) as validation for early stopping
        n_val = max(1, int(0.15 * len(X_train)))
        X_val, y_val = X_train[-n_val:], y_train[-n_val:]
        X_tr, y_tr = X_train[:-n_val], y_train[:-n_val]

        model = _make_xgb(scale_pos_weight=spw)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        raw_probs = model.predict_proba(X_test)[:, 1]
        pred_df = test_df[['season', 'episode', 'castaway_id', 'castaway', 'won_season']].copy()
        pred_df['raw_prob'] = raw_probs
        pred_df = _normalize_probs(pred_df)
        all_preds.append(pred_df)

        # --- metrics for held-out season ---
        # Focus on the final episode snapshot
        final_ep = test_df['episode'].max()
        final_preds = pred_df[pred_df['episode'] == final_ep].copy()
        final_preds = final_preds.sort_values('win_probability', ascending=False).reset_index(drop=True)

        winner_row = final_preds[final_preds['won_season'] == 1]
        winner_rank = winner_row.index[0] + 1 if len(winner_row) > 0 else len(final_preds)
        winner_prob = winner_row['win_probability'].values[0] if len(winner_row) > 0 else 0.0
        top1 = int(winner_rank == 1)

        # AUC on all snapshots (if variation exists)
        try:
            auc = roc_auc_score(y_test, raw_probs)
        except Exception:
            auc = float('nan')

        season_metrics.append({
            'season': held_out,
            'winner_rank_final_ep': winner_rank,
            'winner_prob_final_ep': round(winner_prob, 3),
            'top1_accuracy': top1,
            'auc': round(auc, 3),
            'n_finalists': len(final_preds),
            'best_trees': model.best_iteration,
        })

        status = '✓' if top1 else '✗'
        print(f"  Season {int(held_out):>2d}: {status}  winner rank={winner_rank}/{len(final_preds)}"
              f"  P(win)={winner_prob:.1%}  AUC={auc:.3f}")

    metrics_df = pd.DataFrame(season_metrics)
    preds_df = pd.concat(all_preds, ignore_index=True)

    return metrics_df, preds_df


def train_full_model(feature_df):
    """Train on all completed seasons and save model."""
    X = feature_df[FEATURE_COLS].values
    y = feature_df['won_season'].values

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    spw = n_neg / max(n_pos, 1)

    # Use last 10% as a small eval set (just for early stopping reference)
    n_val = max(50, int(0.10 * len(X)))
    X_tr, y_tr = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]

    model = _make_xgb(scale_pos_weight=spw)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'survivor_model.json')
    model.save_model(model_path)
    print(f"Full model saved → {model_path}  (best_iteration={model.best_iteration})")

    # Feature importance
    importance = {k: float(v) for k, v in zip(FEATURE_COLS, model.feature_importances_)}
    importance_path = os.path.join(MODELS_DIR, 'feature_importance.json')
    with open(importance_path, 'w') as f:
        json.dump(dict(sorted(importance.items(), key=lambda x: -x[1])), f, indent=2)

    return model


def run_training(feature_df):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    metrics_df, preds_df = loso_cross_validation(feature_df)

    # Summary stats
    top1 = metrics_df['top1_accuracy'].mean()
    mean_rank = metrics_df['winner_rank_final_ep'].mean()
    mean_auc = metrics_df['auc'].mean()
    print(f"\nLOSO summary across {len(metrics_df)} seasons:")
    print(f"  Top-1 accuracy (winner ranked #1):  {top1:.1%}")
    print(f"  Mean winner rank at final episode:  {mean_rank:.2f}")
    print(f"  Mean AUC:                           {mean_auc:.3f}")

    metrics_df.to_csv(os.path.join(PROCESSED_DIR, 'loso_results.csv'), index=False)
    preds_df.to_csv(os.path.join(PROCESSED_DIR, 'loso_predictions.csv'), index=False)

    model = train_full_model(feature_df)

    return model, metrics_df, preds_df
