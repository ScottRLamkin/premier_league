"""Evaluate models trained on the CSV produced by MatchStatistics.create_dataset.

Two purposes:
1. Measure what the library's as-shipped dataset achieves on an unseen season.
2. Demonstrate that the `home_points` / `away_points` columns leak the result
   (they are the teams' cumulative points AFTER the match) — anyone training
   on the raw CSV without dropping them gets an inflated, bogus model.

Usage:
    python research/evaluate_library_csv.py /tmp/full_lag5.csv
"""

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

ID_COLS = [
    "game_id", "date", "season", "match_week",
    "home_team_id", "away_team_id", "home_team", "away_team",
]
LEAK_COLS = ["home_points", "away_points"]
TARGET_COLS = ["home_goals", "away_goals"]


def outcome(df):
    return np.select(
        [df.home_goals > df.away_goals, df.home_goals == df.away_goals],
        ["H", "D"], default="A",
    )


def run(df, feature_cols, label):
    train = df[df.season <= "2023-2024"]
    test = df[df.season == "2024-2025"]
    gb = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=400,
        l2_regularization=1.0, random_state=0,
    )
    gb.fit(train[feature_cols], outcome(train))
    proba = gb.predict_proba(test[feature_cols])
    y = outcome(test)
    pred = gb.classes_[np.argmax(proba, axis=1)]
    print(
        f"{label:48s} acc={accuracy_score(y, pred):.4f}  "
        f"log_loss={log_loss(y, proba, labels=list(gb.classes_)):.4f}"
    )


def main():
    df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "/tmp/full_lag5.csv")
    print(f"library CSV: {df.shape[0]} rows x {df.shape[1]} cols")
    clean_features = [
        c for c in df.columns if c not in ID_COLS + LEAK_COLS + TARGET_COLS
    ]
    run(df, clean_features, "library CSV, leak columns DROPPED (honest)")
    run(df, clean_features + LEAK_COLS, "library CSV, leak columns KEPT (bogus)")


if __name__ == "__main__":
    main()
