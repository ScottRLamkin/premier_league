"""xG over/under-performance study on the bundled FBref data.

Questions answered (all leagues, 2017-18 -> 2024-25):
1. Is finishing over-performance (goals - xG) persistent skill or luck?
   Split-half and season-to-season correlations.
2. Does first-half-of-season xG difference predict second-half points better
   than actual goal difference does?

Outputs printed stats, research/results_xg.json and research/xg_analysis.png.

Usage:
    python research/xg_analysis.py [--db /tmp/pl.db]
"""

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load_team_games(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        """SELECT g.id game_id, g.date, g.season, l.name league,
                  gs.team_id, gs.xG,
                  CASE WHEN gs.team_id = g.home_team_id THEN g.home_goals
                       ELSE g.away_goals END goals_for,
                  CASE WHEN gs.team_id = g.home_team_id THEN g.away_goals
                       ELSE g.home_goals END goals_against
           FROM game_stats gs
           JOIN game g ON g.id = gs.game_id
           JOIN league l ON l.id = g.league_id""",
        conn,
        parse_dates=["date"],
    )
    conn.close()
    df["points"] = np.select(
        [df.goals_for > df.goals_against, df.goals_for == df.goals_against],
        [3, 1], default=0)
    # opponent xG against this team (for xG-difference)
    opp = df[["game_id", "team_id", "xG"]].rename(
        columns={"team_id": "opp_id", "xG": "xG_against"})
    df = df.merge(opp, on="game_id")
    df = df[df.team_id != df.opp_id].drop(columns="opp_id")
    return df.sort_values(["team_id", "date"]).reset_index(drop=True)


def split_half(df):
    """Tag each team-season game as first or second half of that team's season."""
    df = df.copy()
    df["game_no"] = df.groupby(["team_id", "season"]).cumcount()
    n_games = df.groupby(["team_id", "season"])["game_no"].transform("max") + 1
    df["half"] = np.where(df["game_no"] < n_games / 2, 1, 2)
    return df


def per_half_aggregates(df):
    agg = (
        df.groupby(["team_id", "season", "half"])
        .agg(games=("points", "size"), points=("points", "sum"),
             goals=("goals_for", "sum"), xg=("xG", "sum"),
             goals_against=("goals_against", "sum"),
             xg_against=("xG_against", "sum"))
        .reset_index()
    )
    agg["overperf_pg"] = (agg.goals - agg.xg) / agg.games
    agg["gdiff_pg"] = (agg.goals - agg.goals_against) / agg.games
    agg["xgdiff_pg"] = (agg.xg - agg.xg_against) / agg.games
    agg["ppg"] = agg.points / agg.games
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/tmp/pl.db")
    args = parser.parse_args()

    df = split_half(load_team_games(args.db))
    agg = per_half_aggregates(df)
    h1 = agg[agg.half == 1].set_index(["team_id", "season"])
    h2 = agg[agg.half == 2].set_index(["team_id", "season"])
    both = h1.join(h2, lsuffix="_h1", rsuffix="_h2", how="inner")
    both = both[(both.games_h1 >= 15) & (both.games_h2 >= 15)]

    results = {"n_team_seasons": len(both)}

    # 1. split-half persistence
    results["split_half_correlations"] = {
        "xG_per_game (skill check)": round(
            float(np.corrcoef(both.xg_h1 / both.games_h1,
                              both.xg_h2 / both.games_h2)[0, 1]), 3),
        "finishing_overperformance (goals - xG)": round(
            float(np.corrcoef(both.overperf_pg_h1, both.overperf_pg_h2)[0, 1]), 3),
    }

    # season-to-season finishing persistence
    season_team = (
        df.groupby(["team_id", "season"])
        .agg(games=("points", "size"), goals=("goals_for", "sum"), xg=("xG", "sum"))
        .reset_index()
    )
    season_team["overperf_pg"] = (season_team.goals - season_team.xg) / season_team.games
    season_team["next_season"] = season_team.season.str[:4].astype(int) + 1
    season_team["next_season"] = (
        season_team.next_season.astype(str) + "-"
        + (season_team.next_season + 1).astype(str))
    pairs = season_team.merge(
        season_team, left_on=["team_id", "next_season"],
        right_on=["team_id", "season"], suffixes=("_t", "_t1"))
    results["season_to_season_overperformance_corr"] = round(
        float(np.corrcoef(pairs.overperf_pg_t, pairs.overperf_pg_t1)[0, 1]), 3)
    results["n_season_pairs"] = len(pairs)

    # 2. which first-half signal predicts second-half points best?
    results["predicting_h2_ppg_from_h1"] = {
        "h1 points per game": round(float(np.corrcoef(both.ppg_h1, both.ppg_h2)[0, 1]), 3),
        "h1 goal difference per game": round(
            float(np.corrcoef(both.gdiff_pg_h1, both.ppg_h2)[0, 1]), 3),
        "h1 xG difference per game": round(
            float(np.corrcoef(both.xgdiff_pg_h1, both.ppg_h2)[0, 1]), 3),
    }

    print(json.dumps(results, indent=2))
    Path("research/results_xg.json").write_text(json.dumps(results, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [
        (both.xg_h1 / both.games_h1, both.xg_h2 / both.games_h2,
         "xG per game: 1st vs 2nd half\n(persistent = skill)"),
        (both.overperf_pg_h1, both.overperf_pg_h2,
         "Finishing (goals − xG) per game:\n1st vs 2nd half"),
        (pairs.overperf_pg_t, pairs.overperf_pg_t1,
         "Finishing (goals − xG) per game:\nseason t vs t+1"),
    ]
    for ax, (x, y, title) in zip(axes, panels):
        ax.scatter(x, y, s=8, alpha=0.4)
        r = np.corrcoef(x, y)[0, 1]
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 10)
        ax.plot(xs, a + b * xs, color="crimson", lw=1.5)
        ax.set_title(f"{title}\nr = {r:.2f}")
        ax.grid(alpha=0.3)
    fig.suptitle("xG persistence vs finishing luck — 6 leagues, 2017-18 to 2024-25")
    fig.tight_layout()
    fig.savefig("research/xg_analysis.png", dpi=130)
    print("written research/results_xg.json and research/xg_analysis.png")


if __name__ == "__main__":
    main()
