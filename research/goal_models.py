"""Goal-based match models: Dixon-Coles-style Poisson and feature Poisson.

Predicts goal rates for both teams, converts a score grid into H/D/A
probabilities (with the Dixon-Coles low-score correction), and evaluates on
the unseen 2024-25 season with accuracy, log loss and ranked probability
score (RPS). Hyperparameters (time-decay xi, low-score rho) are tuned on the
2023-24 season only.

Models:
1. team-strength Poisson (attack/defence dummies, exponential time decay,
   refit monthly during the test season) -- the classic Dixon-Coles recipe,
   with rho profiled on validation outcome log loss instead of inside the
   likelihood (pragmatic simplification).
2. feature Poisson: PoissonRegressor on lag-5 form stats + Elo, one row per
   team-match.
3. blends with the Elo logistic-regression baseline.

Usage:
    python research/goal_models.py [--db /tmp/pl.db]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from build_and_evaluate import build_match_dataset  # noqa: E402

MAX_GOALS = 10
VAL_SEASON = "2023-2024"
TEST_SEASON = "2024-2025"
XI_GRID = [0.0, 0.00125, 0.0025, 0.005]  # per-day decay
RHO_GRID = np.arange(-0.15, 0.11, 0.025)

# condensed per-side form features for the feature-Poisson model
FORM_KEYS = [
    "form_xG", "form_xAG", "form_goals_for", "form_goals_against",
    "form_possession_rate", "form_progressive_passes",
    "form_passes_into_penalty_area", "form_points", "ppg_season",
]


def rps(proba_hda: np.ndarray, outcome_hda: np.ndarray) -> float:
    """Ranked probability score, ordered H > D > A. Lower is better."""
    cum_p = np.cumsum(proba_hda[:, :2], axis=1)
    cum_o = np.cumsum(outcome_hda[:, :2], axis=1)
    return float(np.mean(0.5 * ((cum_p - cum_o) ** 2).sum(axis=1)))


def dc_tau(lam, mu, rho):
    """Dixon-Coles low-score adjustment factors on a (MAX_GOALS+1)^2 grid."""
    tau = np.ones((len(lam), MAX_GOALS + 1, MAX_GOALS + 1))
    tau[:, 0, 0] = 1 - lam * mu * rho
    tau[:, 0, 1] = 1 + lam * rho
    tau[:, 1, 0] = 1 + mu * rho
    tau[:, 1, 1] = 1 - rho
    return np.clip(tau, 1e-10, None)


def grid_probs(lam, mu, rho):
    """(n,) goal rates -> (n, 3) H/D/A probabilities via the score grid."""
    goals = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(goals[None, :], lam[:, None])  # (n, G)
    pa = poisson.pmf(goals[None, :], mu[:, None])
    grid = ph[:, :, None] * pa[:, None, :] * dc_tau(lam, mu, rho)
    grid /= grid.sum(axis=(1, 2), keepdims=True)
    home = np.tril(np.ones((MAX_GOALS + 1, MAX_GOALS + 1)), -1)
    p_home = (grid * home[None]).sum(axis=(1, 2))
    p_draw = np.trace(grid, axis1=1, axis2=2)
    return np.stack([p_home, p_draw, 1 - p_home - p_draw], axis=1)


def long_design(matches, teams, ref_date, xi):
    """Stacked attack/defence dummy design: two rows per match."""
    t_idx = {t: i for i, t in enumerate(teams)}
    n, k = len(matches), len(teams)
    X = np.zeros((2 * n, 2 * k + 1))
    y = np.empty(2 * n)
    h_at = matches["home_team_id"].map(t_idx).to_numpy()
    a_at = matches["away_team_id"].map(t_idx).to_numpy()
    rows = np.arange(n)
    X[rows, h_at] = 1            # home attack
    X[rows, k + a_at] = -1       # away defence
    X[rows, 2 * k] = 1           # home advantage
    y[:n] = matches["home_goals"].to_numpy()
    X[n + rows, a_at] = 1        # away attack
    X[n + rows, k + h_at] = -1   # home defence
    y[n:] = matches["away_goals"].to_numpy()
    days = (ref_date - matches["date"]).dt.days.to_numpy()
    w = np.exp(-xi * days)
    return X, y, np.concatenate([w, w])


def dc_predict(train, predict, ref_date, xi):
    """Fit team-strength Poisson on train, return (lam, mu) for predict."""
    teams = sorted(set(train["home_team_id"]) | set(train["away_team_id"]))
    t_idx = {t: i for i, t in enumerate(teams)}
    X, y, w = long_design(train, teams, ref_date, xi)
    model = PoissonRegressor(alpha=1e-3, max_iter=2000)
    model.fit(X, y, sample_weight=w)

    k = len(teams)
    n = len(predict)
    Xp_h = np.zeros((n, 2 * k + 1))
    Xp_a = np.zeros((n, 2 * k + 1))
    rows = np.arange(n)
    # unseen teams (promoted with no league history) keep all-zero dummies
    # i.e. a league-average team -- documented limitation
    h = predict["home_team_id"].map(t_idx)
    a = predict["away_team_id"].map(t_idx)
    hv, av = h.notna().to_numpy(), a.notna().to_numpy()
    Xp_h[rows[hv], h[hv].astype(int)] = 1
    Xp_h[rows[av], k + a[av].astype(int)] = -1
    Xp_h[:, 2 * k] = 1
    Xp_a[rows[av], a[av].astype(int)] = 1
    Xp_a[rows[hv], k + h[hv].astype(int)] = -1
    return model.predict(Xp_h), model.predict(Xp_a)


def run_dc_season(df, season, xi):
    """Monthly refits per league across one season; returns lam, mu arrays."""
    season_df = df[df["season"] == season]
    out = []
    for league, ldf in season_df.groupby("league"):
        hist = df[(df["league"] == league) & (df["date"] < ldf["date"].min())]
        for month, mdf in ldf.groupby(ldf["date"].dt.to_period("M")):
            ref = mdf["date"].min()
            train = pd.concat([hist, ldf[ldf["date"] < ref]])
            lam, mu = dc_predict(train, mdf, ref, xi)
            out.append(pd.DataFrame({"id": mdf["id"], "lam": lam, "mu": mu}))
    return pd.concat(out)


def outcome_onehot(df):
    return np.stack(
        [
            (df["home_goals"] > df["away_goals"]).to_numpy(float),
            (df["home_goals"] == df["away_goals"]).to_numpy(float),
            (df["home_goals"] < df["away_goals"]).to_numpy(float),
        ],
        axis=1,
    )


def metrics(proba, df):
    onehot = outcome_onehot(df)
    ll = float(-np.mean(np.log(np.clip((proba * onehot).sum(1), 1e-12, None))))
    acc = float(np.mean(proba.argmax(1) == onehot.argmax(1)))
    return {"accuracy": round(acc, 4), "log_loss": round(ll, 4),
            "rps": round(rps(proba, onehot), 4)}


def tune_on_validation(df):
    """Pick (xi, rho) minimising outcome log loss on 2023-24."""
    val = df[df["season"] == VAL_SEASON]
    best = None
    for xi in XI_GRID:
        rates = run_dc_season(df, VAL_SEASON, xi).set_index("id")
        lam = rates.loc[val["id"], "lam"].to_numpy()
        mu = rates.loc[val["id"], "mu"].to_numpy()
        for rho in RHO_GRID:
            ll = metrics(grid_probs(lam, mu, rho), val)["log_loss"]
            if best is None or ll < best[0]:
                best = (ll, xi, float(rho))
    print(f"validation best: log_loss={best[0]:.4f} xi={best[1]} rho={best[2]:.3f}")
    return best[1], best[2]


def feature_poisson(df, feature_keys, rho):
    """Long-format Poisson on form features; returns test H/D/A probs."""
    own = [f"home_{k}" for k in feature_keys]
    opp = [f"away_{k}" for k in feature_keys]

    def stack(d):
        top = d[own + opp + ["elo_diff"]].to_numpy()
        bot = d[opp + own + ["elo_diff"]].to_numpy()
        bot[:, -1] *= -1
        X = np.vstack([np.c_[top, np.ones(len(d))], np.c_[bot, np.zeros(len(d))]])
        y = np.concatenate([d["home_goals"], d["away_goals"]])
        return X, y

    train = df[df["season"] < TEST_SEASON]
    test = df[df["season"] == TEST_SEASON]
    X_tr, y_tr = stack(train)
    model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=2000))
    model.fit(X_tr, y_tr)
    X_te, _ = stack(test)
    pred = model.predict(X_te)
    lam, mu = pred[: len(test)], pred[len(test):]
    return grid_probs(lam, mu, rho), test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/tmp/pl.db")
    parser.add_argument("--out", default="research/results_goal_models.json")
    args = parser.parse_args()

    df, _ = build_match_dataset(args.db)
    df = df.sort_values("date").reset_index(drop=True)

    xi, rho = tune_on_validation(df)

    test = df[df["season"] == TEST_SEASON]
    results = {"tuned": {"xi": xi, "rho": rho}, "n_test": len(test)}

    # Elo logistic-regression baseline (probabilities in H/D/A order)
    train = df[df["season"] < TEST_SEASON]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(train[["elo_diff"]], np.select(
        [train.home_goals > train.away_goals, train.home_goals == train.away_goals],
        ["H", "D"], default="A"))
    order = [list(lr.classes_).index(c) for c in ["H", "D", "A"]]
    p_elo = lr.predict_proba(test[["elo_diff"]])[:, order]

    rates = run_dc_season(df, TEST_SEASON, xi).set_index("id")
    p_dc = grid_probs(rates.loc[test["id"], "lam"].to_numpy(),
                      rates.loc[test["id"], "mu"].to_numpy(), rho)
    p_feat, test_f = feature_poisson(df, FORM_KEYS, rho)
    assert (test_f["id"].to_numpy() == test["id"].to_numpy()).all()

    candidates = {
        "Elo logistic regression (baseline)": p_elo,
        "Dixon-Coles Poisson (decay + rho)": p_dc,
        "Feature Poisson (form + Elo)": p_feat,
        "Blend: DC + feature Poisson": 0.5 * p_dc + 0.5 * p_feat,
        "Blend: DC + feature + Elo-LR": (p_dc + p_feat + p_elo) / 3,
    }
    for league_filter in [None, "Premier League"]:
        mask = np.ones(len(test), bool) if league_filter is None else (
            test["league"] == league_filter).to_numpy()
        scope = league_filter or "all leagues"
        print(f"\n=== test {TEST_SEASON}, {scope} ({mask.sum()} matches) ===")
        results[scope] = {}
        for name, proba in candidates.items():
            m = metrics(proba[mask], test[mask])
            results[scope][name] = m
            print(f"{name:38s} acc={m['accuracy']:.4f}  "
                  f"log_loss={m['log_loss']:.4f}  rps={m['rps']:.4f}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
