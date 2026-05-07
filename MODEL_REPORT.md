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
| **Top-1 Accuracy** (winner ranked #1 at final episode) | **57.1%** |
| Mean winner rank at final episode | 1.78 |
| Median winner rank at final episode | 1 |
| Mean ROC-AUC (all snapshots) | 0.791 |
| Seasons evaluated | 49 |

**Early-game top-1 accuracy (episodes 1–4):** 16.3%
**Late-game top-1 accuracy (final episode):** 57.1%

A random baseline would rank the winner #1 approximately **1/3 = 33%** of the
time (average ~3 finalists). This model substantially exceeds that.

![Rank Distribution](rank_distribution.png)

---

## Feature Importance

The model learned the following feature ranking (by XGBoost gain):

| Rank | Feature | Importance |
|---|---|---|
| 1 | `times_on_the_block` | 0.1405 |
| 2 | `friends_on_jury` | 0.1395 |
| 3 | `votes_received_last_3_eps` | 0.0675 |
| 4 | `total_votes_received` | 0.0651 |
| 5 | `is_returning_player` | 0.0563 |
| 6 | `collar_encoded` | 0.0349 |
| 7 | `age_approx` | 0.0328 |
| 8 | `season_era` | 0.0276 |
| 9 | `was_swap_orphan` | 0.0276 |
| 10 | `gender_female` | 0.0272 |

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
|        1 |                      2 |                  0.078 |               0 | 0.694 |             4 |
|        2 |                      2 |                  0.162 |               0 | 0.863 |             3 |
|        3 |                      2 |                  0.34  |               0 | 0.823 |             4 |
|        4 |                      1 |                  0.398 |               1 | 0.725 |             4 |
|        5 |                      1 |                  0.597 |               1 | 0.906 |             4 |
|        6 |                      2 |                  0.182 |               0 | 0.725 |             4 |
|        7 |                      1 |                  0.674 |               1 | 0.814 |             4 |
|        8 |                      2 |                  0.288 |               0 | 0.751 |             4 |
|        9 |                      1 |                  0.717 |               1 | 0.903 |             4 |
|       10 |                      2 |                  0.144 |               0 | 0.803 |             4 |
|       11 |                      1 |                  0.606 |               1 | 0.913 |             4 |
|       12 |                      1 |                  0.52  |               1 | 0.947 |             3 |
|       13 |                      2 |                  0.474 |               0 | 0.984 |             5 |
|       14 |                      4 |                  0.035 |               0 | 0.376 |             5 |
|       15 |                      3 |                  0.213 |               0 | 0.623 |             4 |
|       16 |                      1 |                  0.685 |               1 | 0.942 |             4 |
|       17 |                      5 |                  0     |               0 | 0.086 |             5 |
|       18 |                      1 |                  0.573 |               1 | 0.986 |             4 |
|       19 |                      5 |                  0.026 |               0 | 0.319 |             5 |
|       20 |                      1 |                  0.451 |               1 | 0.979 |             5 |
|       21 |                      1 |                  0.505 |               1 | 0.842 |             5 |
|       22 |                      2 |                  0.349 |               0 | 0.836 |             4 |
|       23 |                      1 |                  0.701 |               1 | 1     |             5 |
|       24 |                      1 |                  0.483 |               1 | 0.886 |             5 |
|       25 |                      2 |                  0.229 |               0 | 0.813 |             4 |
|       26 |                      1 |                  0.479 |               1 | 0.959 |             5 |
|       27 |                      1 |                  0.689 |               1 | 0.935 |             5 |
|       28 |                      1 |                  0.85  |               1 | 0.993 |             4 |
|       29 |                      1 |                  0.655 |               1 | 0.939 |             5 |
|       30 |                      1 |                  0.311 |               1 | 0.793 |             5 |
|       31 |                      1 |                  0.642 |               1 | 0.968 |             6 |
|       32 |                      1 |                  0.58  |               1 | 0.932 |             4 |
|       33 |                      2 |                  0.368 |               0 | 0.871 |             6 |
|       34 |                      1 |                  0.926 |               1 | 0.949 |             6 |
|       35 |                      1 |                  0.659 |               1 | 0.997 |             5 |
|       36 |                      1 |                  0.789 |               1 | 0.959 |             6 |
|       37 |                      1 |                  0.602 |               1 | 0.89  |             6 |
|       38 |                      6 |                  0.008 |               0 | 0.139 |             6 |
|       39 |                      2 |                  0.202 |               0 | 0.812 |             6 |
|       40 |                      1 |                  0.427 |               1 | 0.784 |             6 |
|       41 |                      3 |                  0.059 |               0 | 0.779 |             5 |
|       42 |                      4 |                  0.067 |               0 | 0.482 |             5 |
|       43 |                      2 |                  0.199 |               0 | 0.74  |             5 |
|       44 |                      1 |                  0.523 |               1 | 0.994 |             5 |
|       45 |                      1 |                  0.346 |               1 | 0.413 |             5 |
|       46 |                      3 |                  0.026 |               0 | 0.555 |             5 |
|       47 |                      1 |                  0.554 |               1 | 0.719 |             4 |
|       48 |                      1 |                  0.387 |               1 | 0.771 |             5 |
|       49 |                      2 |                  0.269 |               0 | 0.87  |             5 |

### Best Predictions (winner ranked #1)

|   season |   winner_rank_final_ep |   winner_prob_final_ep |
|---------:|-----------------------:|-----------------------:|
|        4 |                      1 |                  0.398 |
|        5 |                      1 |                  0.597 |
|        7 |                      1 |                  0.674 |
|        9 |                      1 |                  0.717 |
|       11 |                      1 |                  0.606 |

### Hardest Seasons (winner ranked lowest)

|   season |   winner_rank_final_ep |   winner_prob_final_ep |
|---------:|-----------------------:|-----------------------:|
|       38 |                      6 |                  0.008 |
|       17 |                      5 |                  0     |
|       19 |                      5 |                  0.026 |
|       14 |                      4 |                  0.035 |
|       42 |                      4 |                  0.067 |

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
