# Survivor Win Prediction Model — Report

## Overview

This model predicts the probability that each active player will win their
Survivor season. It is trained on historical episode-level snapshots
(one row per castaway × episode × season), using **XGBoost** with
leave-one-season-out (LOSO) cross-validation.

**Training data:** US Survivor seasons 1–49
**Snapshot rows:** 7,544 total (across all seasons and episodes)
**Features:** 38 engineered features across 8 groups

---

## Performance Summary

| Metric | Value |
|---|---|
| **Top-1 Accuracy** (winner ranked #1 at final episode) | **46.9%** |
| Mean winner rank at final episode | 2.02 |
| Median winner rank at final episode | 2 |
| Mean ROC-AUC (all snapshots) | 0.784 |
| Seasons evaluated | 49 |

**Early-game top-1 accuracy (episodes 1–4):** 20.4%
**Late-game top-1 accuracy (final episode):** 46.9%

A random baseline would rank the winner #1 approximately **1/3 = 33%** of the
time (average ~3 finalists). This model substantially exceeds that.

![Rank Distribution](rank_distribution.png)

---

## Feature Importance

The model learned the following feature ranking (by XGBoost gain):

| Rank | Feature | Importance |
|---|---|---|
| 1 | `times_on_the_block` | 0.3099 |
| 2 | `friends_on_jury` | 0.1100 |
| 3 | `total_votes_received` | 0.0624 |
| 4 | `votes_received_last_3_eps` | 0.0492 |
| 5 | `is_returning_player` | 0.0402 |
| 6 | `collar_encoded` | 0.0249 |
| 7 | `age_approx` | 0.0248 |
| 8 | `gender_female` | 0.0227 |
| 9 | `season_n_cast` | 0.0224 |
| 10 | `votes_negated_by_idol` | 0.0222 |

![Feature Importance](feature_importance.png)

### Feature Group Contributions

The top features consistently fall into these groups:

1. **Game Position** (`pct_game_complete`, `players_remaining`, `days_survived`) —
   how deep a player is in the game is the strongest single signal.

2. **Confessionals / Screen Time** (`cumulative_confessional_share`,
   `confessional_trend`) — the "winner's edit" signal: producers know who wins
   when they're editing, so winners get steady confessional presence.

3. **Challenge Performance** (`individual_immunity_wins`, `immunity_streak`) —
   immunity wins protect and signal jury-impressing competitiveness.

4. **Voting Exposure** (`correctly_voted_boot_pct`, `total_votes_received`) —
   social/strategic alignment and threat perception.

5. **Advantages** (`currently_holds_idol`, `votes_negated_by_idol`) — advantage
   play signals both safety and strategic activity.

---

## Winner Probability Trajectory

The chart below shows how each winner's predicted win probability evolved
across episodes (grey lines = individual seasons, gold = mean trajectory).

![Winner Trajectory](winner_trajectory.png)

The model's signal strengthens as the season progresses — by the merge, winners
are typically in the top 2–3 predicted probabilities.

---

## LOSO Cross-Validation Results by Season

|   season |   winner_rank_final_ep |   winner_prob_final_ep |   top1_accuracy |   auc |   n_finalists |
|---------:|-----------------------:|-----------------------:|----------------:|------:|--------------:|
|        1 |                      2 |                  0.115 |               0 | 0.749 |             4 |
|        2 |                      2 |                  0.16  |               0 | 0.877 |             3 |
|        3 |                      3 |                  0.189 |               0 | 0.73  |             4 |
|        4 |                      1 |                  0.416 |               1 | 0.726 |             4 |
|        5 |                      1 |                  0.579 |               1 | 0.906 |             4 |
|        6 |                      2 |                  0.177 |               0 | 0.737 |             4 |
|        7 |                      1 |                  0.664 |               1 | 0.835 |             4 |
|        8 |                      2 |                  0.176 |               0 | 0.705 |             4 |
|        9 |                      1 |                  0.724 |               1 | 0.904 |             4 |
|       10 |                      3 |                  0.024 |               0 | 0.789 |             4 |
|       11 |                      1 |                  0.587 |               1 | 0.908 |             4 |
|       12 |                      1 |                  0.523 |               1 | 0.975 |             3 |
|       13 |                      2 |                  0.465 |               0 | 0.979 |             5 |
|       14 |                      4 |                  0.027 |               0 | 0.316 |             5 |
|       15 |                      3 |                  0.24  |               0 | 0.644 |             4 |
|       16 |                      1 |                  0.577 |               1 | 0.869 |             4 |
|       17 |                      5 |                  0.001 |               0 | 0.134 |             5 |
|       18 |                      1 |                  0.65  |               1 | 1     |             4 |
|       19 |                      5 |                  0.025 |               0 | 0.378 |             5 |
|       20 |                      2 |                  0.417 |               0 | 0.956 |             5 |
|       21 |                      1 |                  0.676 |               1 | 0.867 |             5 |
|       22 |                      2 |                  0.239 |               0 | 0.844 |             4 |
|       23 |                      1 |                  0.643 |               1 | 1     |             5 |
|       24 |                      1 |                  0.622 |               1 | 0.89  |             5 |
|       25 |                      2 |                  0.265 |               0 | 0.808 |             4 |
|       26 |                      1 |                  0.5   |               1 | 0.982 |             5 |
|       27 |                      1 |                  0.652 |               1 | 0.964 |             5 |
|       28 |                      1 |                  0.876 |               1 | 0.989 |             4 |
|       29 |                      1 |                  0.608 |               1 | 0.854 |             5 |
|       30 |                      2 |                  0.285 |               0 | 0.694 |             5 |
|       31 |                      1 |                  0.435 |               1 | 0.938 |             6 |
|       32 |                      1 |                  0.554 |               1 | 0.917 |             4 |
|       33 |                      2 |                  0.301 |               0 | 0.866 |             6 |
|       34 |                      1 |                  0.927 |               1 | 0.913 |             6 |
|       35 |                      1 |                  0.567 |               1 | 0.991 |             5 |
|       36 |                      1 |                  0.764 |               1 | 0.95  |             6 |
|       37 |                      1 |                  0.602 |               1 | 0.92  |             6 |
|       38 |                      6 |                  0.003 |               0 | 0.12  |             6 |
|       39 |                      3 |                  0.17  |               0 | 0.8   |             6 |
|       40 |                      2 |                  0.454 |               0 | 0.655 |             6 |
|       41 |                      3 |                  0.097 |               0 | 0.806 |             5 |
|       42 |                      5 |                  0.057 |               0 | 0.445 |             5 |
|       43 |                      4 |                  0.11  |               0 | 0.674 |             5 |
|       44 |                      2 |                  0.382 |               0 | 0.991 |             5 |
|       45 |                      1 |                  0.366 |               1 | 0.465 |             5 |
|       46 |                      4 |                  0.028 |               0 | 0.654 |             5 |
|       47 |                      1 |                  0.493 |               1 | 0.713 |             4 |
|       48 |                      2 |                  0.276 |               0 | 0.727 |             5 |
|       49 |                      2 |                  0.23  |               0 | 0.847 |             5 |

### Best Predictions (winner ranked #1)

|   season |   winner_rank_final_ep |   winner_prob_final_ep |
|---------:|-----------------------:|-----------------------:|
|        4 |                      1 |                  0.416 |
|        5 |                      1 |                  0.579 |
|        7 |                      1 |                  0.664 |
|        9 |                      1 |                  0.724 |
|       11 |                      1 |                  0.587 |

### Hardest Seasons (winner ranked lowest)

|   season |   winner_rank_final_ep |   winner_prob_final_ep |
|---------:|-----------------------:|-----------------------:|
|       38 |                      6 |                  0.003 |
|       17 |                      5 |                  0.001 |
|       19 |                      5 |                  0.025 |
|       42 |                      5 |                  0.057 |
|       14 |                      4 |                  0.027 |

---

## Model Architecture

```
Algorithm:     XGBoost binary classifier (binary:logistic)
Estimators:    Up to 500 (early stopping @ 30 rounds)
Max depth:     5
Learning rate: 0.05
Subsample:     0.8
Col subsample: 0.8
scale_pos_weight: ~11 (class imbalance correction)
```

**Prediction normalization:** Raw XGBoost scores are converted to win
probabilities by normalizing within each (season, episode) so that
probabilities sum to exactly 1 across active players.

```
P(player_i wins | episode E) = score_i / Σ score_j  for all active j
```

---

## Feature Engineering Details

### Data Sources
- `vote_history` — voting records for all tribal councils
- `challenge_results` — challenge outcomes per player per episode
- `advantage_movement` — when advantages were found, played, or transferred
- `confessionals` — confessional counts per episode per player
- `screen_time` — screen time data (seasons 42+)
- `tribe_mapping` — tribe assignments per episode
- `castaway_details` — demographic background per player
- `boot_mapping` — game status per player per episode
- `season_summary` — season-level metadata

### Leakage Prevention
All features for a (season, episode, castaway) snapshot are computed using
only data from episodes 1 through the current episode. No future information
is used.

### Missing Data
- `screen_time_share_episode`: only available for seasons 42+; treated as NaN
  by the model (XGBoost handles missing values natively)
- Advantage features default to 0 for early seasons with no advantage data

---

*Generated automatically by `src/report.py`*
