"""
FiveThirtyEight-style win-probability trajectory plot for a Survivor season.

Usage:
    python -m src.plot [--season 50] [--out reports/trajectory_s50.png]

The chart shows every castaway's normalized win probability across all
episodes, overlaid on the same axes. Eliminated players fade to grey;
active players are coloured and annotated at the right edge.
"""

import os
import argparse
import textwrap

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.lines
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reports')
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')

# ── FiveThirtyEight colour palette ────────────────────────────────────────────
_538_COLORS = [
    '#008FD5',  # 538 blue
    '#FC4F30',  # 538 red
    '#E5AE38',  # 538 gold
    '#6D904F',  # 538 green
    '#8B8B8B',  # medium grey
    '#810F7C',  # purple
    '#F781BF',  # pink
    '#A65628',  # brown
    '#4DAF4A',  # lime
    '#377EB8',  # steel blue
    '#FF7F00',  # orange
    '#984EA3',  # violet
    '#E41A1C',  # crimson
    '#A6CEE3',  # light blue
    '#1F78B4',  # mid blue
    '#B2DF8A',  # light green
    '#33A02C',  # dark green
    '#FB9A99',  # salmon
    '#FDBF6F',  # peach
    '#CAB2D6',  # lavender
    '#FFFF99',  # pale yellow
    '#B15928',  # sienna
    '#66C2A5',  # teal
    '#FC8D62',  # coral
]

_FTE_BG = '#F0F0F0'
_FTE_GRID = '#FFFFFF'
_ELIM_COLOR = '#CCCCCC'
_ELIM_ALPHA = 0.45


def _fte_axes(ax):
    """Apply FiveThirtyEight styling to an axes object."""
    ax.set_facecolor(_FTE_BG)
    ax.figure.patch.set_facecolor(_FTE_BG)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#999999')
    ax.tick_params(axis='both', which='both', length=0, labelsize=10,
                   colors='#555555')
    ax.yaxis.set_tick_params(pad=4)
    ax.xaxis.set_tick_params(pad=4)
    ax.grid(axis='y', color=_FTE_GRID, linewidth=1.2, zorder=0)
    ax.grid(axis='x', color=_FTE_GRID, linewidth=0.6, linestyle=':', zorder=0)
    ax.set_axisbelow(True)


def plot_trajectory(traj_df, season=50, title=None, out_path=None):
    """
    Draw a FiveThirtyEight-style win-probability trajectory chart.

    Parameters
    ----------
    traj_df : DataFrame
        Columns: episode, castaway, castaway_id, win_probability
    season : int
    title : str, optional
    out_path : str, optional
    """
    traj_df = traj_df.copy()
    traj_df['episode'] = traj_df['episode'].astype(int)

    all_players = sorted(traj_df['castaway'].unique())
    latest_ep = traj_df['episode'].max()
    active_players = set(traj_df[traj_df['episode'] == latest_ep]['castaway'].tolist())
    elim_players = [p for p in all_players if p not in active_players]

    # Sort active players by final probability (descending) to assign top colours
    final_probs = (traj_df[traj_df['episode'] == latest_ep]
                   .set_index('castaway')['win_probability']
                   .sort_values(ascending=False))
    active_sorted = final_probs.index.tolist()

    # Assign colours: active → distinct palette, eliminated → grey
    color_map = {}
    for i, player in enumerate(active_sorted):
        color_map[player] = _538_COLORS[i % len(_538_COLORS)]
    for player in elim_players:
        color_map[player] = _ELIM_COLOR

    episodes = sorted(traj_df['episode'].unique())
    n_eps = len(episodes)

    # ── Figure layout ──────────────────────────────────────────────────────────
    # Right-side margin for player name labels
    label_width = 0.18
    fig_w = max(11, n_eps * 0.85 + 3.5)
    fig_h = 7.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.07, right=1 - label_width - 0.01,
                        top=0.88, bottom=0.10)
    _fte_axes(ax)

    # ── Draw eliminated players first (behind) ─────────────────────────────────
    for player in elim_players:
        pdata = traj_df[traj_df['castaway'] == player].sort_values('episode')
        ax.plot(pdata['episode'], pdata['win_probability'] * 100,
                color=_ELIM_COLOR, linewidth=1.4, alpha=_ELIM_ALPHA,
                solid_capstyle='round', zorder=1)
        # Dot at elimination episode
        ax.scatter(pdata['episode'].iloc[-1],
                   pdata['win_probability'].iloc[-1] * 100,
                   color=_ELIM_COLOR, s=22, zorder=2, alpha=_ELIM_ALPHA)

    # ── Draw active players on top ─────────────────────────────────────────────
    label_entries = []   # (rank, player, color, last_ep, raw_y_pct)
    for rank, player in enumerate(active_sorted):
        pdata = traj_df[traj_df['castaway'] == player].sort_values('episode')
        color = color_map[player]
        lw = 2.8 if rank == 0 else (2.2 if rank <= 2 else 1.8)

        ax.plot(pdata['episode'], pdata['win_probability'] * 100,
                color=color, linewidth=lw,
                solid_capstyle='round', zorder=3 + rank)

        last_ep = pdata['episode'].iloc[-1]
        last_prob = pdata['win_probability'].iloc[-1] * 100
        ax.scatter(last_ep, last_prob, color=color, s=45, zorder=10 + rank,
                   edgecolors='white', linewidths=0.8)
        label_entries.append((rank, player, color, last_ep, last_prob))

    # ── Right-margin label strip with leader lines ─────────────────────────────
    # Place labels at evenly-spaced positions in the right margin, ordered by
    # descending final probability. Leader lines connect each label to the dot
    # at the end of the player's line.
    fig.canvas.draw()   # materialise transforms

    n_active = len(label_entries)
    # Fraction of figure height available for labels (leave 8% top/bottom margin)
    margin_top_frac = 0.82      # figure-fraction y for topmost label
    margin_bot_frac = 0.14      # figure-fraction y for bottommost label
    if n_active == 1:
        label_y_fracs = [(margin_top_frac + margin_bot_frac) / 2]
    else:
        step = (margin_top_frac - margin_bot_frac) / (n_active - 1)
        label_y_fracs = [margin_top_frac - i * step for i in range(n_active)]

    # x position for label text (fixed, just inside right edge)
    label_x_frac = 1 - label_width + 0.012

    label_entries_sorted = sorted(label_entries, key=lambda e: -e[4])  # high → low

    for i, (rank, player, color, last_ep, raw_y) in enumerate(label_entries_sorted):
        lbl_y = label_y_fracs[i]

        # Convert the dot's data coords to figure-fraction
        x_fig_dot, y_fig_dot = ax.transData.transform((last_ep, raw_y))
        x_dot_frac, y_dot_frac = fig.transFigure.inverted().transform((x_fig_dot, y_fig_dot))

        # Leader line: from the dot (x just right of axes boundary) to label y
        ax_x1_frac = fig.subplotpars.right + 0.004
        # Draw a short horizontal stub from dot, then angle to label
        leader = matplotlib.lines.Line2D(
            [x_dot_frac + 0.003, ax_x1_frac, label_x_frac - 0.005],
            [y_dot_frac,         y_dot_frac, lbl_y],
            transform=fig.transFigure,
            color=color, linewidth=0.9, alpha=0.55,
            solid_capstyle='round', zorder=20,
        )
        fig.add_artist(leader)

        # Label text
        prob_str = f'{raw_y:.0f}%'
        label_text = f'{player}  {prob_str}'
        fig.text(label_x_frac, lbl_y, label_text,
                 va='center', ha='left',
                 fontsize=10 if rank == 0 else (9 if rank <= 2 else 8.5),
                 fontweight='bold' if rank == 0 else 'normal',
                 color=color,
                 path_effects=[pe.withStroke(linewidth=2.5,
                                             foreground=_FTE_BG)])

    # ── Axes formatting ────────────────────────────────────────────────────────
    ax.set_xlim(episodes[0] - 0.3, episodes[-1] + 0.3)
    ax.set_ylim(-2, 102)
    ax.set_xticks(episodes)
    ax.set_xticklabels([f'Ep {e}' for e in episodes], fontsize=9, color='#555555')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax.set_ylabel('Win Probability', fontsize=10, color='#555555', labelpad=8)

    # ── Episode dividers (thin vertical lines at each ep) ─────────────────────
    for ep in episodes:
        ax.axvline(ep, color='#DDDDDD', linewidth=0.6, zorder=0)

    # ── Eliminated player legend entry ────────────────────────────────────────
    elim_handle = Line2D([0], [0], color=_ELIM_COLOR, linewidth=1.8,
                         alpha=_ELIM_ALPHA, label='Eliminated')
    ax.legend(handles=[elim_handle], loc='upper left', fontsize=8.5,
              frameon=False, labelcolor='#888888')

    # ── Title & subtitle ──────────────────────────────────────────────────────
    if title is None:
        title = f'Survivor Season {season} — Win Probability Tracker'
    sub = (f'Episode {int(episodes[-1])} of Season {season}  ·  '
           'Probabilities normalized to sum to 100% across active players  ·  '
           'Model trained on Seasons 1–49')

    fig.text(0.07, 0.955, title, ha='left', va='bottom',
             fontsize=15, fontweight='bold', color='#333333')
    fig.text(0.07, 0.925, sub, ha='left', va='bottom',
             fontsize=8, color='#777777')

    # ── FTE logo / watermark strip ─────────────────────────────────────────────
    fig.text(0.99, 0.01, 'survivor-model', ha='right', va='bottom',
             fontsize=7, color='#AAAAAA', style='italic')

    os.makedirs(REPORTS_DIR, exist_ok=True)
    if out_path is None:
        out_path = os.path.join(REPORTS_DIR, f'trajectory_s{season}.png')

    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=_FTE_BG, edgecolor='none')
    plt.close(fig)
    print(f"Plot saved → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=int, default=50)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    traj_path = os.path.join(PROCESSED_DIR, f'trajectory_s{args.season}.csv')
    if not os.path.exists(traj_path):
        # Build trajectory on the fly
        from src.data_loader import load_all
        from src.predict import predict_trajectory
        dfs = load_all()
        traj_df = predict_trajectory(dfs, season=args.season)
    else:
        traj_df = pd.read_csv(traj_path)

    plot_trajectory(traj_df, season=args.season, out_path=args.out)


if __name__ == '__main__':
    main()
