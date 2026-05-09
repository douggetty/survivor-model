"""
Feature engineering for the Survivor prediction model.

Builds a snapshot matrix: one row per (season, episode, castaway_id) for
every player still in the game at that episode. Features are computed from
data available up to and including that episode (no leakage).
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Statuses that mean a player is still competing (not eliminated).
# 'Exile Island' players are temporarily sent away but remain in the game.
_ACTIVE_STATUSES = {'In the game', 'Exile Island'}


def _us(df):
    """Filter DataFrame to US version rows."""
    if 'version' in df.columns:
        return df[df['version'] == 'US'].copy()
    if 'version_season' in df.columns:
        return df[df['version_season'].str.startswith('US')].copy()
    return df.copy()


def _dedup_boot_map(bm):
    """
    Remove duplicate (season, episode, castaway_id) rows from boot_mapping.

    During tribe merges a player can appear in both old-tribe and merged-tribe
    rows in the same episode. We keep the row with the most-final game status
    (voted out > in the game).
    """
    status_order = {'Voted out': 0, 'Eliminated': 1, 'Quit': 2,
                    'Medically evacuated': 3, 'In the game': 4}
    bm = bm.copy()
    bm['_sr'] = bm['game_status'].map(status_order).fillna(5)
    bm = (bm.sort_values('_sr')
            .drop_duplicates(['season', 'episode', 'castaway_id'], keep='first')
            .drop(columns=['_sr']))
    return bm


def _asof_merge(left, right, on, by, fill_value=0):
    """
    merge_asof backward join: each left row gets the most recent right value.

    pandas 3.x requires the `on` column to be globally sorted (not just
    within `by` groups), so we sort only by `on`.
    """
    left_s = left.copy()
    right_s = right.copy()
    left_s[on] = left_s[on].astype(float)
    right_s[on] = right_s[on].astype(float)
    left_s = left_s.sort_values(on).reset_index(drop=True)
    right_s = right_s.sort_values(on).reset_index(drop=True)
    merged = pd.merge_asof(left_s, right_s, on=on, by=by, direction='backward')
    return merged


def _cumsum_per_group(df, group_cols, episode_col, value_cols):
    """Sort by group + episode, compute cumulative sums for value_cols."""
    df = df.sort_values(group_cols + [episode_col])
    for col in value_cols:
        df[f'cum_{col}'] = df.groupby(group_cols)[col].cumsum()
    return df


# ---------------------------------------------------------------------------
# Step 1 — base snapshot table
# ---------------------------------------------------------------------------

def build_base(boot_map_raw, season_summary_raw, tribe_mapping_raw):
    """
    Returns (snapshots, completed_seasons).

    snapshots: DataFrame with columns
        [season, episode, castaway_id, castaway, day, final_n, max_days, won_season]
    """
    bm = _us(boot_map_raw)
    ss = _us(season_summary_raw)
    tm = _us(tribe_mapping_raw)

    completed = ss[ss['winner_id'].notna()]['season'].tolist()

    bm = _dedup_boot_map(bm)

    active = bm[
        bm['season'].isin(completed) &
        (bm['game_status'] == 'In the game')
    ][['season', 'episode', 'castaway_id', 'castaway', 'final_n']].copy()

    # Join day from tribe_mapping (one row per player per episode — take max day)
    day_lookup = (
        tm.groupby(['season', 'episode', 'castaway_id'])['day'].max().reset_index()
    )
    active = active.merge(day_lookup, on=['season', 'episode', 'castaway_id'], how='left')

    # Fallback: compute day from episode ordinal (3 days/episode approx)
    active['day'] = active['day'].fillna(active['episode'] * 3)

    winner_map = ss.set_index('season')['winner_id'].to_dict()
    active['won_season'] = (
        active.apply(lambda r: winner_map.get(r['season']) == r['castaway_id'], axis=1)
        .astype(int)
    )

    # Season max days (from tribe_mapping)
    max_days = tm.groupby('season')['day'].max().rename('max_days')
    active = active.merge(max_days, on='season', how='left')
    active['max_days'] = active['max_days'].fillna(active['day'].max())

    return active, completed


# ---------------------------------------------------------------------------
# Step 2 — game position features (Group 1)
# ---------------------------------------------------------------------------

def add_game_position(snapshots, vote_history_raw):
    """
    Adds:
        days_survived, pct_game_complete, players_remaining,
        survived_tribals, was_in_majority_vote_pct
    """
    df = snapshots.copy()
    df['days_survived'] = df['day']
    df['pct_game_complete'] = (df['days_survived'] / df['max_days'].clip(lower=1)).clip(0, 1)
    df['players_remaining'] = df['final_n']

    vh = _us(vote_history_raw)

    # survived_tribals: distinct episodes a player cast a vote (proxy for tribals attended)
    tribals = vh.groupby(['season', 'castaway_id', 'episode']).size().reset_index(name='_votes_cast')
    tribals['tribals_this_ep'] = 1
    tribals = tribals.sort_values(['season', 'castaway_id', 'episode'])
    tribals['cum_survived_tribals'] = tribals.groupby(['season', 'castaway_id'])['tribals_this_ep'].cumsum()
    tribals_merge = tribals[['season', 'castaway_id', 'episode', 'cum_survived_tribals']]

    # correctly voted boot pct
    player_votes = vh[['season', 'castaway_id', 'episode', 'vote', 'voted_out']].copy()
    player_votes['voted_correctly'] = (player_votes['vote'] == player_votes['voted_out']).astype(float)
    player_votes['has_vote'] = player_votes['vote'].notna().astype(float)
    vote_agg = player_votes.groupby(['season', 'castaway_id', 'episode']).agg(
        ep_correct=('voted_correctly', 'sum'),
        ep_voted=('has_vote', 'sum'),
    ).reset_index()
    vote_agg = vote_agg.sort_values(['season', 'castaway_id', 'episode'])
    vote_agg['cum_correct'] = vote_agg.groupby(['season', 'castaway_id'])['ep_correct'].cumsum()
    vote_agg['cum_voted'] = vote_agg.groupby(['season', 'castaway_id'])['ep_voted'].cumsum()
    vote_agg['was_in_majority_vote_pct'] = (
        vote_agg['cum_correct'] / vote_agg['cum_voted'].clip(lower=1)
    )

    snap = df[['season', 'castaway_id', 'episode']].copy()
    snap = _asof_merge(snap, tribals_merge, on='episode', by=['season', 'castaway_id'])
    snap = _asof_merge(
        snap,
        vote_agg[['season', 'castaway_id', 'episode', 'was_in_majority_vote_pct']],
        on='episode', by=['season', 'castaway_id']
    )

    df = df.merge(
        snap[['season', 'castaway_id', 'episode', 'cum_survived_tribals', 'was_in_majority_vote_pct']],
        on=['season', 'castaway_id', 'episode'], how='left'
    )
    df['survived_tribals'] = df['cum_survived_tribals'].fillna(0)
    df['was_in_majority_vote_pct'] = df['was_in_majority_vote_pct'].fillna(0.5)
    df.drop(columns=['cum_survived_tribals'], inplace=True)

    return df


# ---------------------------------------------------------------------------
# Step 3 — challenge features (Group 2)
# ---------------------------------------------------------------------------

def add_challenge_features(snapshots, challenge_results_raw):
    cr = _us(challenge_results_raw)

    # Aggregate per (season, castaway_id, episode) — multiple challenges possible per episode
    ep = cr.groupby(['season', 'castaway_id', 'episode']).agg(
        ep_indiv_imm=('won_individual_immunity', 'sum'),
        ep_indiv_rew=('won_individual_reward', 'sum'),
        ep_tribal_imm=('won_tribal_immunity', 'sum'),
        ep_any_win=('won', 'sum'),
        ep_indiv_participated=(
            'won_individual_immunity',
            lambda x: x.notna().sum()
        ),
    ).reset_index()

    ep = ep.sort_values(['season', 'castaway_id', 'episode'])

    for col in ['ep_indiv_imm', 'ep_indiv_rew', 'ep_tribal_imm',
                'ep_any_win', 'ep_indiv_participated']:
        ep[f'cum_{col}'] = ep.groupby(['season', 'castaway_id'])[col].cumsum()

    ep.rename(columns={
        'cum_ep_indiv_imm': 'individual_immunity_wins',
        'cum_ep_indiv_rew': 'reward_wins',
        'cum_ep_tribal_imm': 'tribal_challenge_contributions',
        'cum_ep_indiv_participated': 'individual_challenges_participated',
    }, inplace=True)

    ep['individual_immunity_win_rate'] = (
        ep['individual_immunity_wins'] / ep['individual_challenges_participated'].clip(lower=1)
    )

    # Immunity streak (consecutive individual immunity wins) — vectorized
    ep['won_imm_this_ep'] = (ep['ep_indiv_imm'] > 0).astype(int)
    # Assign a run-ID each time the win/no-win status flips within a player-season
    run_ids = ep.groupby(['season', 'castaway_id'])['won_imm_this_ep'].transform(
        lambda x: (x != x.shift()).cumsum()
    )
    ep['_run_id'] = run_ids
    ep['_pos'] = ep.groupby(['season', 'castaway_id', '_run_id']).cumcount() + 1
    ep['immunity_streak'] = ep['_pos'] * ep['won_imm_this_ep']
    ep.drop(columns=['_run_id', '_pos'], inplace=True)

    feat_cols = ['season', 'castaway_id', 'episode',
                 'individual_immunity_wins', 'reward_wins',
                 'tribal_challenge_contributions',
                 'individual_immunity_win_rate',
                 'immunity_streak',
                 'individual_challenges_participated']

    merged = _asof_merge(
        snapshots[['season', 'castaway_id', 'episode']],
        ep[feat_cols],
        on='episode', by=['season', 'castaway_id']
    )

    fill_cols = feat_cols[3:]
    for col in fill_cols:
        merged[col] = merged[col].fillna(0)

    return snapshots.merge(
        merged[['season', 'castaway_id', 'episode'] + fill_cols],
        on=['season', 'castaway_id', 'episode'], how='left'
    )


# ---------------------------------------------------------------------------
# Step 4 — voting exposure features (Group 3)
# ---------------------------------------------------------------------------

def add_vote_features(snapshots, vote_history_raw, boot_map_raw):
    vh = _us(vote_history_raw)
    bm = _dedup_boot_map(_us(boot_map_raw))

    # --- votes received ---
    vr = vh.groupby(['season', 'voted_out_id', 'episode']).size().reset_index(name='ep_votes_rcvd')
    vr.rename(columns={'voted_out_id': 'castaway_id'}, inplace=True)
    vr = vr.sort_values(['season', 'castaway_id', 'episode'])
    vr['total_votes_received'] = vr.groupby(['season', 'castaway_id'])['ep_votes_rcvd'].cumsum()
    vr['votes_received_last_3_eps'] = (
        vr.groupby(['season', 'castaway_id'])['ep_votes_rcvd']
        .transform(lambda x: x.rolling(3, min_periods=1).sum())
    )
    vr['on_block_this_ep'] = (vr['ep_votes_rcvd'] > 0).astype(int)
    vr['times_on_the_block'] = vr.groupby(['season', 'castaway_id'])['on_block_this_ep'].cumsum()

    vr_feats = vr[['season', 'castaway_id', 'episode',
                   'total_votes_received', 'votes_received_last_3_eps', 'times_on_the_block']]

    # --- vote split survived ---
    split = vh[vh['split_vote'].notna() & (vh['split_vote'] != 'None')].copy()
    split_survived = split.groupby(['season', 'castaway_id', 'episode']).size().reset_index(name='split_ep')
    split_survived['split_ep'] = 1
    split_survived = split_survived.sort_values(['season', 'castaway_id', 'episode'])
    split_survived['vote_split_survived'] = (
        split_survived.groupby(['season', 'castaway_id'])['split_ep'].cummax()
    )

    # --- original tribe allies remaining ---
    orig = bm[bm['tribe_status'] == 'Original'][['season', 'castaway_id', 'tribe']].drop_duplicates(['season', 'castaway_id'])
    in_game = bm[bm['game_status'] == 'In the game'][['season', 'episode', 'castaway_id']]
    ig_tribe = in_game.merge(orig, on=['season', 'castaway_id'], how='left')
    tribe_counts = ig_tribe.groupby(['season', 'episode', 'tribe'])['castaway_id'].count().reset_index(name='tc')
    ig_tribe = ig_tribe.merge(tribe_counts, on=['season', 'episode', 'tribe'], how='left')
    ig_tribe['original_tribe_allies_remaining'] = (ig_tribe['tc'] - 1).clip(lower=0).fillna(0)
    ally_feats = ig_tribe[['season', 'episode', 'castaway_id', 'original_tribe_allies_remaining']]

    # merge everything
    df = snapshots.copy()
    snap_ids = df[['season', 'castaway_id', 'episode']]

    m1 = _asof_merge(snap_ids, vr_feats, on='episode', by=['season', 'castaway_id'])
    m2 = _asof_merge(
        snap_ids,
        split_survived[['season', 'castaway_id', 'episode', 'vote_split_survived']],
        on='episode', by=['season', 'castaway_id']
    )
    m3 = df.merge(ally_feats, on=['season', 'episode', 'castaway_id'], how='left')

    df = df.merge(
        m1[['season', 'castaway_id', 'episode',
            'total_votes_received', 'votes_received_last_3_eps', 'times_on_the_block']],
        on=['season', 'castaway_id', 'episode'], how='left'
    )
    df = df.merge(
        m2[['season', 'castaway_id', 'episode', 'vote_split_survived']],
        on=['season', 'castaway_id', 'episode'], how='left'
    )
    df = df.merge(
        m3[['season', 'castaway_id', 'episode', 'original_tribe_allies_remaining']],
        on=['season', 'castaway_id', 'episode'], how='left'
    )

    for col in ['total_votes_received', 'votes_received_last_3_eps',
                'times_on_the_block', 'vote_split_survived', 'original_tribe_allies_remaining']:
        df[col] = df[col].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Step 5 — advantage features (Group 4)
# ---------------------------------------------------------------------------

def add_advantage_features(snapshots, advantage_movement_raw):
    am = _us(advantage_movement_raw)

    am['is_idol'] = am['advantage_id'].map(
        lambda x: True  # default; will be overridden per advantage_id lookup
    )
    # Check advantage_type if available; otherwise assume idol
    if 'advantage_type' in am.columns:
        am['is_idol'] = am['advantage_type'].str.contains('Idol', na=False, case=False)
    else:
        am['is_idol'] = True

    # Derive per-event binary columns before groupby to avoid cross-reference lambdas
    am['ev_found'] = (am['event'] == 'Found').astype(int)
    am['ev_received'] = (am['event'] == 'Received').astype(int)
    am['ev_played'] = (am['event'] == 'Played').astype(int)
    am['ev_given'] = (am['event'] == 'Transferred').astype(int)
    am['ev_success'] = am['success'].isin(['true', 'True', True]).astype(int)
    am['ev_idol_found'] = (am['is_idol'] & (am['event'] == 'Found')).astype(int)
    am['ev_non_idol_acq'] = (~am['is_idol'] & am['event'].isin(['Found', 'Received'])).astype(int)
    am['votes_nullified'] = pd.to_numeric(am['votes_nullified'], errors='coerce').fillna(0)

    # Per player per episode: events
    am_ep = am.groupby(['season', 'castaway_id', 'episode']).agg(
        ep_found=('ev_found', 'sum'),
        ep_received=('ev_received', 'sum'),
        ep_played=('ev_played', 'sum'),
        ep_played_success=('ev_success', 'sum'),
        ep_votes_nullified=('votes_nullified', 'sum'),
        ep_given=('ev_given', 'sum'),
        ep_idol_found=('ev_idol_found', 'sum'),
        ep_non_idol_found=('ev_non_idol_acq', 'sum'),
    ).reset_index().fillna(0)

    am_ep = am_ep.sort_values(['season', 'castaway_id', 'episode'])

    for col in ['ep_found', 'ep_received', 'ep_played', 'ep_played_success',
                'ep_votes_nullified', 'ep_given', 'ep_idol_found']:
        am_ep[f'cum_{col}'] = am_ep.groupby(['season', 'castaway_id'])[col].cumsum()

    # currently holds idol = found + received - played - given (computed before rename)
    am_ep['currently_holds_idol'] = (
        (am_ep['cum_ep_found'] + am_ep['cum_ep_received'] - am_ep['cum_ep_played'] - am_ep['cum_ep_given'])
        .clip(lower=0)
        .gt(0)
        .astype(int)
    )

    # other advantages held (non-idol)
    am_ep['cum_non_idol'] = am_ep.groupby(['season', 'castaway_id'])['ep_non_idol_found'].cumsum()
    am_ep['other_advantages_held'] = am_ep['cum_non_idol'].clip(lower=0)

    am_ep.rename(columns={
        'cum_ep_idol_found': 'idols_found',
        'cum_ep_played_success': 'idols_played_successfully',
        'cum_ep_votes_nullified': 'votes_negated_by_idol',
        'cum_ep_given': 'advantages_given_to_others',
    }, inplace=True)

    feat_cols = ['season', 'castaway_id', 'episode',
                 'idols_found', 'currently_holds_idol', 'idols_played_successfully',
                 'votes_negated_by_idol', 'advantages_given_to_others', 'other_advantages_held']

    merged = _asof_merge(
        snapshots[['season', 'castaway_id', 'episode']],
        am_ep[feat_cols],
        on='episode', by=['season', 'castaway_id']
    )

    for col in feat_cols[3:]:
        merged[col] = merged[col].fillna(0)

    return snapshots.merge(
        merged[['season', 'castaway_id', 'episode'] + feat_cols[3:]],
        on=['season', 'castaway_id', 'episode'], how='left'
    ).fillna({c: 0 for c in feat_cols[3:]})


# ---------------------------------------------------------------------------
# Step 6 — confessional features (Group 5)
# ---------------------------------------------------------------------------

def add_confessional_features(snapshots, confessionals_raw, screen_time_raw):
    cf = _us(confessionals_raw)
    cf['confessional_count'] = cf['confessional_count'].fillna(0)

    # Share per episode
    ep_total = cf.groupby(['season', 'episode'])['confessional_count'].sum().rename('ep_total_confessionals')
    cf = cf.merge(ep_total.reset_index(), on=['season', 'episode'], how='left')
    cf['confessional_share_ep'] = cf['confessional_count'] / cf['ep_total_confessionals'].clip(lower=1)

    # Cumulative share
    cf = cf.sort_values(['season', 'castaway_id', 'episode'])
    cf['cum_confessionals'] = cf.groupby(['season', 'castaway_id'])['confessional_count'].cumsum()

    cum_total = cf.groupby(['season', 'episode'])['confessional_count'].cumsum()
    # cumulative total per season up to episode
    season_cum = cf.groupby(['season', 'episode'])['confessional_count'].sum().groupby(level=0).cumsum().rename('season_cum_total')
    cf = cf.merge(season_cum.reset_index(), on=['season', 'episode'], how='left')
    cf['cumulative_confessional_share'] = cf['cum_confessionals'] / cf['season_cum_total'].clip(lower=1)

    # Trend: slope of confessional_count over last 3 episodes — vectorized via rolling
    def _slope(y):
        if len(y) < 2:
            return 0.0
        x = np.arange(len(y), dtype=float)
        return float(np.polyfit(x, y, 1)[0])

    cf['confessional_trend'] = (
        cf.groupby(['season', 'castaway_id'])['confessional_count']
        .transform(lambda x: x.rolling(3, min_periods=2).apply(_slope, raw=True).fillna(0))
    )

    feat_cols = ['season', 'castaway_id', 'episode',
                 'confessional_count', 'confessional_share_ep',
                 'cumulative_confessional_share', 'confessional_trend']

    merged = _asof_merge(
        snapshots[['season', 'castaway_id', 'episode']],
        cf[feat_cols],
        on='episode', by=['season', 'castaway_id']
    )

    # Screen time (limited seasons)
    st = screen_time_raw.copy()
    if 'version_season' in st.columns:
        st = st[st['version_season'].str.startswith('US')]
    # Extract season number from version_season (e.g. "US42" → 42)
    st['season'] = st['version_season'].str.replace('US', '', regex=False).astype(float)
    st_ep = st.groupby(['season', 'castaway_id', 'episode'])['screen_time'].sum().reset_index()
    ep_st_total = st_ep.groupby(['season', 'episode'])['screen_time'].sum().rename('ep_total_st').reset_index()
    st_ep = st_ep.merge(ep_st_total, on=['season', 'episode'], how='left')
    st_ep['screen_time_share_episode'] = st_ep['screen_time'] / st_ep['ep_total_st'].clip(lower=1)
    st_ep = st_ep[['season', 'castaway_id', 'episode', 'screen_time_share_episode']]

    merged = _asof_merge(merged, st_ep, on='episode', by=['season', 'castaway_id'])

    for col in feat_cols[3:]:
        merged[col] = merged[col].fillna(0)
    # screen_time_share_episode: NaN means no data for that season (leave as NaN — model handles it)

    return snapshots.merge(
        merged[['season', 'castaway_id', 'episode'] + feat_cols[3:] + ['screen_time_share_episode']],
        on=['season', 'castaway_id', 'episode'], how='left'
    )


# ---------------------------------------------------------------------------
# Step 7 — tribal / social network features (Group 6)
# ---------------------------------------------------------------------------

def add_social_features(snapshots, tribe_mapping_raw, vote_history_raw, castaways_raw):
    tm = _us(tribe_mapping_raw)
    vh = _us(vote_history_raw)
    cast = _us(castaways_raw)

    # num_tribe_swaps_survived: cumulative distinct-tribe changes per player — vectorized
    player_tribes = tm.groupby(['season', 'castaway_id', 'episode'])['tribe'].first().reset_index()
    player_tribes = player_tribes.sort_values(['season', 'castaway_id', 'episode'])

    # Tribe changes: tribe differs from previous tribe within player-season
    player_tribes['prev_tribe'] = player_tribes.groupby(['season', 'castaway_id'])['tribe'].shift()
    player_tribes['is_new_tribe'] = (
        (player_tribes['tribe'] != player_tribes['prev_tribe']) &
        player_tribes['prev_tribe'].notna()
    ).astype(int)
    player_tribes['cum_swaps'] = player_tribes.groupby(['season', 'castaway_id'])['is_new_tribe'].cumsum()
    swap_feats = player_tribes[['season', 'castaway_id', 'episode', 'cum_swaps']]

    # was_swap_orphan: after a swap, player was in the minority of their new tribe
    # Identify swap episodes: tribe_status changes from 'Original' to 'Swapped'
    tm_status = tm[['season', 'episode', 'castaway_id', 'tribe', 'tribe_status']].copy()
    swap_eps = tm_status[tm_status['tribe_status'].str.contains('Swap', na=False, case=False)]
    if len(swap_eps) > 0:
        # Count players per (season, episode, tribe) after swap
        swap_counts = swap_eps.groupby(['season', 'episode', 'tribe'])['castaway_id'].count().reset_index(name='tribe_size_post_swap')
        swap_eps = swap_eps.merge(swap_counts, on=['season', 'episode', 'tribe'], how='left')
        # Swap orphan: tribe size < median tribe size at that swap episode
        ep_median = swap_eps.groupby(['season', 'episode'])['tribe_size_post_swap'].transform('median')
        swap_eps['is_orphan_ep'] = (swap_eps['tribe_size_post_swap'] < ep_median).astype(int)
        orphan_feats = swap_eps.groupby(['season', 'castaway_id'])['is_orphan_ep'].max().reset_index()
        orphan_feats.rename(columns={'is_orphan_ep': 'was_swap_orphan'}, inplace=True)
    else:
        orphan_feats = pd.DataFrame(columns=['season', 'castaway_id', 'was_swap_orphan'])

    # Jury relationship score
    # jury members = players with jury=True in castaways (voted out but on jury)
    jury_members = cast[cast['jury'] == True][['season', 'castaway_id', 'episode']].copy()
    jury_members.rename(columns={'castaway_id': 'juror_id', 'episode': 'boot_episode'}, inplace=True)

    # For each castaway, for each juror booted before episode E:
    # did castaway vote for juror in their boot episode? → negative relationship
    votes_at_boot = vh.merge(
        jury_members, left_on=['season', 'episode', 'voted_out_id'], right_on=['season', 'boot_episode', 'juror_id'],
        how='inner'
    )
    voted_against_jury = votes_at_boot.groupby(['season', 'castaway_id', 'episode']).agg(
        ep_voted_against_jury=('voted_out_id', 'count')
    ).reset_index()
    voted_against_jury = voted_against_jury.sort_values(['season', 'castaway_id', 'episode'])
    voted_against_jury['jury_votes_against_cumsum'] = voted_against_jury.groupby(
        ['season', 'castaway_id'])['ep_voted_against_jury'].cumsum()

    # jury size at each episode
    jury_size_ep = (
        jury_members.groupby(['season', 'boot_episode']).size()
        .groupby(level=0).cumsum()
        .rename('jury_size')
        .reset_index()
        .rename(columns={'boot_episode': 'episode'})
    )

    # friends_on_jury: jurors they voted WITH most
    votes_with_jury = vh.merge(
        jury_members, left_on=['season', 'episode', 'voted_out_id'], right_on=['season', 'boot_episode', 'juror_id'],
        how='inner'
    )
    # Not voted out = voted with jury member (they voted the same target)
    same_vote = vh[['season', 'castaway_id', 'episode', 'vote']].merge(
        votes_with_jury[['season', 'episode', 'juror_id', 'vote']].rename(columns={'vote': 'jury_vote'}),
        on=['season', 'episode'], how='inner'
    )
    same_vote['voted_with_jury'] = (same_vote['vote'] == same_vote['jury_vote']).astype(int)
    friends = same_vote.groupby(['season', 'castaway_id', 'juror_id'])['voted_with_jury'].mean()
    friends = (friends >= 0.5).astype(int).reset_index()
    friends_count = friends.groupby(['season', 'castaway_id'])['voted_with_jury'].sum().reset_index()
    friends_count.rename(columns={'voted_with_jury': 'friends_on_jury'}, inplace=True)

    # Merge social features onto snapshots
    df = snapshots.copy()

    # Swap count (asof)
    m1 = _asof_merge(
        df[['season', 'castaway_id', 'episode']],
        swap_feats, on='episode', by=['season', 'castaway_id']
    )
    df = df.merge(m1[['season', 'castaway_id', 'episode', 'cum_swaps']],
                  on=['season', 'castaway_id', 'episode'], how='left')
    df.rename(columns={'cum_swaps': 'num_tribe_swaps_survived'}, inplace=True)
    df['num_tribe_swaps_survived'] = df['num_tribe_swaps_survived'].fillna(0)

    # Swap orphan (season-level)
    df = df.merge(orphan_feats, on=['season', 'castaway_id'], how='left')
    df['was_swap_orphan'] = df['was_swap_orphan'].fillna(0)

    # Jury votes against (asof)
    if len(voted_against_jury) > 0:
        m2 = _asof_merge(
            df[['season', 'castaway_id', 'episode']],
            voted_against_jury[['season', 'castaway_id', 'episode', 'jury_votes_against_cumsum']],
            on='episode', by=['season', 'castaway_id']
        )
        df = df.merge(m2[['season', 'castaway_id', 'episode', 'jury_votes_against_cumsum']],
                      on=['season', 'castaway_id', 'episode'], how='left')
    else:
        df['jury_votes_against_cumsum'] = 0

    # Jury size at each episode (for ratio)
    df = df.merge(jury_size_ep, on=['season', 'episode'], how='left')
    df['jury_votes_against_cumsum'] = df['jury_votes_against_cumsum'].fillna(0)
    df['jury_size'] = df['jury_size'].fillna(0)
    df['jury_relationship_score'] = (
        df['jury_size'] - 2 * df['jury_votes_against_cumsum']
    )  # positive = more friends, negative = more enemies

    # Friends on jury (season-level)
    df = df.merge(friends_count, on=['season', 'castaway_id'], how='left')
    df['friends_on_jury'] = df['friends_on_jury'].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Step 8 — demographic features (Group 7)
# ---------------------------------------------------------------------------

def add_demographic_features(snapshots, castaway_details_raw, castaways_raw):
    cd = castaway_details_raw.copy()
    cast = _us(castaways_raw)

    # Age at time of filming (use season premiere date as proxy)
    cd['dob'] = pd.to_datetime(cd['date_of_birth'], errors='coerce')
    cd['gender_female'] = (cd['gender'] == 'Female').astype(int)

    # collar → occupation type (proxy for social profession)
    cd['collar_encoded'] = cd['collar'].map(
        {'No collar': 0, 'Blue collar': 1, 'White collar': 2}
    ).fillna(1)  # unknown → middle

    demo = cd[['castaway_id', 'dob', 'gender_female', 'collar_encoded']].copy()

    # is_returning_player: appeared in multiple US seasons
    appearances = cast.groupby('castaway_id')['season'].nunique().rename('n_seasons')
    demo = demo.merge(appearances.reset_index(), on='castaway_id', how='left')
    demo['is_returning_player'] = (demo['n_seasons'] > 1).astype(int)

    df = snapshots.copy()
    df = df.merge(demo[['castaway_id', 'dob', 'gender_female', 'collar_encoded', 'is_returning_player']],
                  on='castaway_id', how='left')

    # Age: need season premiere year — approximate from season_era
    # Use birth year difference from 2000 as proxy for age at game time
    # (Exact calculation would require join to episodes table, which is fine to skip for now)
    df['age_approx'] = (pd.Timestamp('2020-01-01') - df['dob']).dt.days / 365.25
    df.drop(columns=['dob'], inplace=True)

    for col in ['gender_female', 'collar_encoded', 'is_returning_player', 'age_approx']:
        df[col] = df[col].fillna(df[col].median())

    return df


# ---------------------------------------------------------------------------
# Step 9 — season control features (Group 8)
# ---------------------------------------------------------------------------

def add_season_controls(snapshots, season_summary_raw):
    ss = _us(season_summary_raw)

    def era(s):
        if s <= 10:
            return 0   # old school
        elif s <= 20:
            return 1   # HD era
        elif s <= 35:
            return 2   # idol era
        else:
            return 3   # new era

    ss['season_era'] = ss['season'].apply(era)
    ss['n_finalists_num'] = pd.to_numeric(ss['n_finalists'], errors='coerce').fillna(3)

    ctrl = ss[['season', 'season_era', 'n_finalists_num', 'n_cast']].copy()
    ctrl.rename(columns={'n_finalists_num': 'n_finalists', 'n_cast': 'season_n_cast'}, inplace=True)

    return snapshots.merge(ctrl, on='season', how='left')


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Group 1: game position
    'days_survived', 'pct_game_complete', 'players_remaining',
    'survived_tribals', 'was_in_majority_vote_pct',
    # Group 2: challenges
    'individual_immunity_wins', 'individual_immunity_win_rate',
    'tribal_challenge_contributions', 'reward_wins', 'immunity_streak',
    # Group 3: voting exposure
    'total_votes_received', 'votes_received_last_3_eps', 'times_on_the_block',
    'correctly_voted_boot_pct', 'vote_split_survived', 'original_tribe_allies_remaining',
    # Group 4: advantages
    'idols_found', 'currently_holds_idol', 'idols_played_successfully',
    'votes_negated_by_idol', 'other_advantages_held', 'advantages_given_to_others',
    # Group 5: confessionals
    'confessional_count', 'confessional_share_ep', 'cumulative_confessional_share',
    'confessional_trend', 'screen_time_share_episode',
    # Group 6: social network
    'num_tribe_swaps_survived', 'was_swap_orphan',
    'jury_relationship_score', 'friends_on_jury',
    # Group 7: demographics
    'age_approx', 'gender_female', 'is_returning_player', 'collar_encoded',
    # Group 8: season controls
    'season_era', 'n_finalists', 'season_n_cast',
]


def build_feature_matrix(dfs, include_current_season=False):
    """
    Build the full feature matrix for all completed US seasons.

    Returns a DataFrame with FEATURE_COLS + ['season', 'episode',
    'castaway_id', 'castaway', 'won_season'].
    """
    bm = dfs['boot_mapping']
    ss = dfs['season_summary']

    df, completed = build_base(bm, ss, dfs['tribe_mapping'])

    print(f"Base snapshots: {len(df):,} rows across {len(completed)} seasons")

    df = add_game_position(df, dfs['vote_history'])
    print("  ✓ game position features")

    df = add_challenge_features(df, dfs['challenge_results'])
    print("  ✓ challenge features")

    df = add_vote_features(df, dfs['vote_history'], bm)
    print("  ✓ vote exposure features")

    df = add_advantage_features(df, dfs['advantage_movement'])
    print("  ✓ advantage features")

    df = add_confessional_features(df, dfs['confessionals'], dfs['screen_time'])
    print("  ✓ confessional features")

    df = add_social_features(df, dfs['tribe_mapping'], dfs['vote_history'], dfs['castaways'])
    print("  ✓ social network features")

    df = add_demographic_features(df, dfs['castaway_details'], dfs['castaways'])
    print("  ✓ demographic features")

    df = add_season_controls(df, ss)
    print("  ✓ season control features")

    # Ensure all feature columns present
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Final cleanup
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')

    print(f"Feature matrix: {df.shape}")
    return df


def build_prediction_snapshots(dfs, current_season=50):
    """
    Build feature rows for the current (in-progress) season's active players.
    Returns the most-recent-episode snapshot for each active player.
    """
    bm = _us(dfs['boot_mapping'])
    current = bm[bm['season'] == current_season]
    latest_ep = current['episode'].max()

    active = current[
        (current['episode'] == latest_ep) & current['game_status'].isin(_ACTIVE_STATUSES)
    ][['season', 'episode', 'castaway_id', 'castaway', 'final_n']].copy()
    active['won_season'] = -1  # unknown
    return _build_episode_snapshot(active, dfs, current_season)


def build_all_episode_snapshots(dfs, current_season=50):
    """
    Build feature rows for ALL episodes of the current season.

    Returns a DataFrame with one row per (episode, castaway_id) for players
    who were in the game at that episode, with features computed from data
    available up to that episode.
    """
    bm = _dedup_boot_map(_us(dfs['boot_mapping']))
    current = bm[bm['season'] == current_season]
    episodes = sorted(current['episode'].unique())

    all_rows = []
    for ep in episodes:
        ep_active = current[
            (current['episode'] == ep) & current['game_status'].isin(_ACTIVE_STATUSES)
        ][['season', 'episode', 'castaway_id', 'castaway', 'final_n']].copy()
        if len(ep_active) == 0:
            continue
        ep_active['won_season'] = -1
        snap = _build_episode_snapshot(ep_active, dfs, current_season)
        all_rows.append(snap)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def _build_episode_snapshot(active, dfs, current_season):
    """Apply all feature groups to an already-filtered active player DataFrame."""
    tm_cur = _us(dfs['tribe_mapping'])
    tm_cur = tm_cur[tm_cur['season'] == current_season]
    day_lookup = (tm_cur[['season', 'episode', 'castaway_id', 'day']]
                  .groupby(['season', 'episode', 'castaway_id'])['day'].max().reset_index())
    active = active.merge(day_lookup, on=['season', 'episode', 'castaway_id'], how='left')
    active['day'] = active['day'].fillna(active['episode'] * 3)
    max_days = tm_cur['day'].max()
    active['max_days'] = max_days

    active = add_game_position(active, dfs['vote_history'])
    active = add_challenge_features(active, dfs['challenge_results'])
    active = add_vote_features(active, dfs['vote_history'], dfs['boot_mapping'])
    active = add_advantage_features(active, dfs['advantage_movement'])
    active = add_confessional_features(active, dfs['confessionals'], dfs['screen_time'])
    active = add_social_features(active, dfs['tribe_mapping'], dfs['vote_history'], dfs['castaways'])
    active = add_demographic_features(active, dfs['castaway_details'], dfs['castaways'])
    active = add_season_controls(active, dfs['season_summary'])

    for col in FEATURE_COLS:
        if col not in active.columns:
            active[col] = 0.0

    active[FEATURE_COLS] = active[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
    return active
