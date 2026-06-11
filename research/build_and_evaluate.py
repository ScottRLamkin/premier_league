"""Premier League match-outcome prediction research.

Builds an ML dataset directly from the bundled SQLite data (same lag-average
idea as MatchStatistics.create_dataset, plus Elo / form / rest-day features),
then trains and evaluates models with a strict time-based split.

Usage:
    python research/build_and_evaluate.py [--db /tmp/pl.db] [--league "Premier League"]

The bundled premier_league.sql must already be loaded into the SQLite db file
(see README in this folder).
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LAG = 5
ELO_K = 20.0
ELO_HOME_ADV = 60.0

META_COLS = ["id", "game_id", "team_id"]


def load_tables(db_path: str):
    conn = sqlite3.connect(db_path)
    games = pd.read_sql(
        """SELECT g.id, g.date, g.season, g.match_week, g.league_id, l.name AS league,
                  g.home_team_id, g.away_team_id, g.home_goals, g.away_goals
           FROM game g JOIN league l ON g.league_id = l.id""",
        conn,
        parse_dates=["date"],
    )
    stats = pd.read_sql("SELECT * FROM game_stats", conn)
    conn.close()
    # null save_percentage (keeper faced no shots) would cascade through
    # rolling windows and drop ~15% of rows; impute with the median
    stats["save_percentage"] = stats["save_percentage"].fillna(
        stats["save_percentage"].median()
    )
    games = games.sort_values("date").reset_index(drop=True)
    return games, stats


def build_long_form(games: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team) with that team's match stats + result info."""
    stat_cols = [c for c in stats.columns if c not in META_COLS]
    long_df = stats.merge(
        games[
            [
                "id",
                "date",
                "season",
                "league",
                "home_team_id",
                "away_team_id",
                "home_goals",
                "away_goals",
            ]
        ],
        left_on="game_id",
        right_on="id",
        how="inner",
    )
    is_home = long_df["team_id"] == long_df["home_team_id"]
    long_df["goals_for"] = np.where(is_home, long_df["home_goals"], long_df["away_goals"])
    long_df["goals_against"] = np.where(
        is_home, long_df["away_goals"], long_df["home_goals"]
    )
    long_df["points"] = np.select(
        [long_df["goals_for"] > long_df["goals_against"],
         long_df["goals_for"] == long_df["goals_against"]],
        [3, 1],
        default=0,
    )
    long_df["was_home"] = is_home.astype(int)
    return long_df, stat_cols


def add_rolling_features(long_df: pd.DataFrame, stat_cols: list) -> pd.DataFrame:
    """Lagged same-season rolling means (shift(1) => strictly pre-match info)."""
    long_df = long_df.sort_values(["team_id", "date"]).reset_index(drop=True)
    roll_cols = stat_cols + ["goals_for", "goals_against", "points", "was_home"]
    grouped = long_df.groupby(["team_id", "season"], sort=False)
    rolled = (
        grouped[roll_cols]
        .apply(lambda g: g.shift(1).rolling(LAG, min_periods=LAG).mean())
        .reset_index(drop=True)
    )
    rolled.columns = [f"form_{c}" for c in rolled.columns]
    # season-to-date points per game (expanding, pre-match)
    long_df["ppg_season"] = grouped["points"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    long_df["rest_days"] = grouped["date"].transform(lambda s: s.diff().dt.days)
    return pd.concat([long_df, rolled], axis=1)


def add_elo(games: pd.DataFrame) -> pd.DataFrame:
    """Sequential Elo per team (global across leagues, K=20, home adv 60)."""
    ratings: dict = {}
    home_elo, away_elo = [], []
    for row in games.itertuples():
        rh = ratings.get(row.home_team_id, 1500.0)
        ra = ratings.get(row.away_team_id, 1500.0)
        home_elo.append(rh)
        away_elo.append(ra)
        exp_home = 1.0 / (1.0 + 10 ** ((ra - (rh + ELO_HOME_ADV)) / 400.0))
        score = 1.0 if row.home_goals > row.away_goals else (
            0.5 if row.home_goals == row.away_goals else 0.0
        )
        delta = ELO_K * (score - exp_home)
        ratings[row.home_team_id] = rh + delta
        ratings[row.away_team_id] = ra - delta
    games = games.copy()
    games["home_elo"] = home_elo
    games["away_elo"] = away_elo
    games["elo_diff"] = games["home_elo"] - games["away_elo"]
    return games


def build_match_dataset(db_path: str) -> pd.DataFrame:
    games, stats = load_tables(db_path)
    games = add_elo(games)
    long_df, stat_cols = build_long_form(games, stats)
    long_df = add_rolling_features(long_df, stat_cols)

    feat_cols = (
        [f"form_{c}" for c in stat_cols]
        + ["form_goals_for", "form_goals_against", "form_points", "form_was_home"]
        + ["ppg_season", "rest_days"]
    )
    keep = ["game_id", "team_id"] + feat_cols
    home = long_df[keep].rename(columns={c: f"home_{c}" for c in feat_cols})
    away = long_df[keep].rename(columns={c: f"away_{c}" for c in feat_cols})

    df = games.merge(
        home, left_on=["id", "home_team_id"], right_on=["game_id", "team_id"]
    ).merge(
        away,
        left_on=["id", "away_team_id"],
        right_on=["game_id", "team_id"],
        suffixes=("", "_away"),
    )
    df["outcome"] = np.select(
        [df["home_goals"] > df["away_goals"], df["home_goals"] == df["away_goals"]],
        ["H", "D"],
        default="A",
    )
    feature_columns = (
        [f"home_{c}" for c in feat_cols]
        + [f"away_{c}" for c in feat_cols]
        + ["home_elo", "away_elo", "elo_diff"]
    )
    df = df.dropna(subset=[c for c in feature_columns if c != "rest_days"])
    return df, feature_columns


def evaluate(name, y_true, proba, classes, results):
    pred = classes[np.argmax(proba, axis=1)]
    acc = accuracy_score(y_true, pred)
    ll = log_loss(y_true, proba, labels=list(classes))
    cm = confusion_matrix(y_true, pred, labels=list(classes))
    results[name] = {
        "accuracy": round(float(acc), 4),
        "log_loss": round(float(ll), 4),
        "confusion_matrix": {"labels": list(classes), "matrix": cm.tolist()},
    }
    print(f"{name:38s} acc={acc:.4f}  log_loss={ll:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/tmp/pl.db")
    parser.add_argument("--league", default=None, help="restrict eval league")
    parser.add_argument("--out", default="research/results.json")
    args = parser.parse_args()

    df, feature_columns = build_match_dataset(args.db)
    print(f"usable matches after lag filtering: {len(df)}")

    train = df[df["season"] <= "2022-2023"]
    val = df[df["season"] == "2023-2024"]
    test = df[df["season"] == "2024-2025"]
    if args.league:
        val = val[val["league"] == args.league]
        test = test[test["league"] == args.league]
    print(f"train={len(train)}  val(23-24)={len(val)}  test(24-25)={len(test)}")

    X_tr, y_tr = train[feature_columns].fillna(7), train["outcome"]
    X_te, y_te = test[feature_columns].fillna(7), test["outcome"]

    classes = np.array(sorted(y_tr.unique()))  # A, D, H
    results = {"n_train": len(train), "n_test": len(test), "league": args.league or "all"}

    # Baseline 1: always home win
    proba = np.tile(
        (y_tr.value_counts(normalize=True).reindex(classes).values), (len(y_te), 1)
    )
    evaluate("baseline: class priors (home-biased)", y_te, proba, classes, results)

    # Baseline 2: Elo-only logistic regression
    elo_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    elo_lr.fit(X_tr[["elo_diff"]], y_tr)
    evaluate(
        "Elo-diff only logistic regression",
        y_te,
        elo_lr.predict_proba(X_te[["elo_diff"]]),
        classes,
        results,
    )

    # Full logistic regression
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.05))
    lr.fit(X_tr, y_tr)
    evaluate("LogReg, all 199 form features", y_te, lr.predict_proba(X_te), classes, results)

    # Gradient boosting, tuned lightly on the validation season
    gb = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=400,
        l2_regularization=1.0, random_state=0,
    )
    gb.fit(X_tr, y_tr)
    evaluate("HistGradientBoosting", y_te, gb.predict_proba(X_te), classes, results)

    # Ensemble of LR + GB
    proba_ens = 0.5 * lr.predict_proba(X_te) + 0.5 * gb.predict_proba(X_te)
    evaluate("Ensemble (LR + GB)", y_te, proba_ens, classes, results)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
