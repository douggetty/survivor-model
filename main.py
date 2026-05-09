"""
Main entry point: build features, run LOSO CV, train full model, generate report.

Usage:
    python main.py                 # full pipeline
    python main.py --predict-only  # skip training, just predict current season
    python main.py --season 50     # specify current season for prediction
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_all
from src.features import build_feature_matrix
from src.train import run_training
from src.predict import predict_current_season, predict_trajectory
from src.report import generate_report
from src.plot import plot_trajectory


def main():
    parser = argparse.ArgumentParser(description='Survivor prediction model')
    parser.add_argument('--predict-only', action='store_true',
                        help='Skip training, run prediction only')
    parser.add_argument('--season', type=int, default=50,
                        help='Current season to predict (default: 50)')
    parser.add_argument('--no-predict', action='store_true',
                        help='Skip prediction step')
    args = parser.parse_args()

    dfs = load_all()

    if not args.predict_only:
        print("\n=== Building feature matrix ===")
        feature_df = build_feature_matrix(dfs)

        os.makedirs('data/processed', exist_ok=True)
        feature_df.to_csv('data/processed/feature_matrix.csv', index=False)
        print(f"Feature matrix saved: {feature_df.shape}")

        print("\n=== Running LOSO cross-validation & training full model ===")
        model, metrics_df, preds_df = run_training(feature_df)

        print("\n=== Generating report ===")
        generate_report(model)

    if not args.no_predict:
        print(f"\n=== Predicting Season {args.season} ===")
        try:
            predict_current_season(dfs, season=args.season)
        except Exception as e:
            print(f"Prediction failed: {e}")

        print(f"\n=== Building trajectory for Season {args.season} ===")
        try:
            traj_df = predict_trajectory(dfs, season=args.season)
            if traj_df is not None:
                plot_trajectory(traj_df, season=args.season)
        except Exception as e:
            print(f"Trajectory failed: {e}")


if __name__ == '__main__':
    main()
