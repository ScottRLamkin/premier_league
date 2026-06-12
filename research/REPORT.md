# Premier League Data Library — Research Report

**Question:** How well does this repo's data gathering work, and how good a
predictive model can be built from its data?

**Short answer:** The library works as advertised and ships a genuinely useful,
clean database (~18k matches, 6 leagues, 2017-18 → May 2025, ~190 stats per
match). An honest match-outcome model built on it reaches **~51% accuracy /
~1.00 log loss** on a fully held-out season — squarely in the published
academic range (50–55%) and a little behind bookmaker closing odds (~53% /
~0.95). One important caveat: the ML CSV the library exports contains two
columns that **leak the match result**, and a naive model trained on the raw
CSV looks much better than it really is.

---

## 1. What the library actually is

- `MatchStatistics` — the core asset. Scrapes FBref match reports into a
  SQLite DB (`league`, `team`, `game`, `game_stats` tables) and exports a
  lag-averaged ML dataset via `create_dataset()`.
- `RankingTable`, `PlayerSeasonLeaders`, `Transfers` — lighter scrapers
  (worldfootball.net) for standings, top scorers/assisters, transfer windows.
- Flask API + AWS Lambda wrappers around the above.

## 2. Does the data gathering work?

**Bundled data (verified):** the package ships a 16 MB SQL dump that
initializes automatically on first use.

| | |
|---|---|
| Matches | 17,945 (each with 2 full stat lines = 35,890 rows) |
| Leagues | Premier League, La Liga, Serie A, Bundesliga, Ligue 1, EFL Championship |
| Coverage | 2017-18 season → **May 2, 2025** (2024-25 nearly complete) |
| Stats per team-match | ~93 columns: xG, xAG, PSxG, shots/passes/tackles/touches/carries split by position group (FW/MF/DF), keeper stats, cards, fouls |

**Quality checks (all passed):** every game has exactly two stat rows; the only
null column is `save_percentage` (~3% of rows — games where the keeper faced no
shots on target, a legitimate null); 13 duplicate (home, away, season) pairs
(replayed/rescheduled fixtures — negligible).

**Pipeline test:** `MatchStatistics()` initialized the DB and
`create_dataset(lag=5)` exported the full **15,549-row × 198-column** dataset in
**92 seconds**, no errors. The lag logic is correctly leak-free for the stat
features: each row contains only averages of *strictly earlier* same-season
games.

**Live scraping (`update_data_set`)** could not be tested from this sandbox —
fbref.com and worldfootball.net are outside this environment's network
allowlist (the 403s observed are from the egress proxy, not the sites). The
bundled data extending to May 2025 shows the pipeline worked recently, and the
code has sensible protections (4 s rate limit, flat-file page cache, hard exit
on HTTP 429). Real-world risk: FBref's anti-bot measures have tightened over
time, so expect occasional 403s and very slow full refreshes (4 s/page ×
thousands of match pages).

### Issues found

1. **🔴 Data leakage in the exported CSV.** `home_points` / `away_points` are
   the teams' *cumulative season points after the match* (verified: matchweek-1
   rows show exactly 3/1/0 matching each result). `create_dataset()` moves only
   `home_goals`/`away_goals` to the target position and leaves these two in the
   feature block. Measured effect (gradient boosting, test = 2024-25 season):

   | Training columns | Accuracy | Log loss |
   |---|---|---|
   | Leak columns dropped (honest) | 49.9% | 1.011 |
   | Leak columns kept (bogus) | **56.8%** | **0.926** |

   "Better than the bookmakers" is the classic signature of leakage. Anyone
   using the CSV must drop these two columns (along with the id/name columns).

2. **Lag filtering discards early-season games.** A row requires `lag` prior
   *same-season* games for both teams, so the default `lag=10` silently drops
   roughly the first quarter of every season. `lag=5` keeps 15,549 of 17,945
   matches (87%).

3. **Performance/code quality (minor).** `__calculate_team_stats` recomputes
   the full weighted aggregate inside its accumulation loop (O(lag²) per team
   per game); results are still correct because only the final iteration's
   value is returned.

## 3. How good a predictive model can you build?

Setup: 3-way match outcome (home/draw/away), strict time split — train on
2017-18 → 2022-23, light tuning on 2023-24, **report on the unseen 2024-25
season**. Features: lag-5 same-season rolling means of all ~93 stats for both
sides (replicating the library's logic, cross-validated to produce the
identical 15,549 rows), plus season points-per-game, rest days, and a
sequential Elo rating. Code: `build_and_evaluate.py`.

**Test season 2024-25, all leagues (1,730 matches):**

| Model | Accuracy | Log loss |
|---|---|---|
| Class priors (always home win) | 43.5% | 1.075 |
| **Elo difference only** (logistic regression) | **51.2%** | **1.001** |
| LogReg, all 199 form features | 50.6% | 1.021 |
| HistGradientBoosting, all features | 50.6% | 1.005 |
| LR + GB ensemble | 50.9% | 1.007 |
| *Reference: bookmaker closing odds (literature)* | *~53%* | *~0.95–0.96* |

Premier-League-only test (260 matches) gives the same picture: Elo-only 51.2% /
0.996.

**Three findings worth internalizing:**

1. **A single Elo-difference feature matches or beats 199 detailed stat
   features.** Almost all the predictable signal in football outcomes is "which
   team is stronger + home advantage." The granular FBref stats (touches,
   carries, aerials by position group...) add essentially nothing for 3-way
   outcome classification — they're collinear, noisy proxies for strength.
2. **No model ever predicts a draw.** Draws are ~26% of matches but are never
   the single most likely outcome, so argmax classifiers ignore them entirely.
   This is the known hard core of the problem; it's why serious work models
   *probabilities* (or goals directly) rather than class labels.
3. **The ceiling is low and that's not a data problem.** Football is
   low-scoring and high-variance; published research and betting-market
   efficiency both put the realistic ceiling around 53–55% accuracy /
   ~0.95 log loss. Landing at 51% / 1.00 from this dataset means the data is
   good — the remaining gap to the bookmakers is mostly information you don't
   have (lineups, injuries, motivation, market flow).

## 4. Recommendations for your data science project

- **Use this library for its database, treat the scraper as a bonus.** The
  bundled SQLite DB is the real asset; `update_data_set()` is best-effort
  against FBref's anti-bot defenses.
- **Drop `home_points`/`away_points`** (and ids/team names) before training on
  `create_dataset()` output.
- **Beat the accuracy framing.** Predict goals with a Poisson / Dixon-Coles
  model or predict calibrated outcome probabilities; evaluate with log loss,
  Brier score, or ranked probability score against bookmaker odds
  (football-data.co.uk has free historical odds CSVs back to the 1990s — also a
  way to extend results-only history far beyond 2017).
- **Where this dataset can actually shine** is not outcome classification:
  xG-vs-results over/under-performance and regression-to-mean studies, draw
  modeling, style-of-play clustering from the positional stat splits,
  transfer-window effects (combine with the `Transfers` class), or a
  value-betting backtest (model probability vs. market price).

---

# Part 2 — Goal models and xG analysis

Follow-up implementing the recommendations above: probability-first goal
models and the xG persistence study.

## 5. Goal-based models (Dixon-Coles and feature Poisson)

Instead of classifying H/D/A directly, predict each team's goal rate, build
the full score grid (with the Dixon-Coles low-score correction) and read off
outcome probabilities. Hyperparameters — time-decay `xi` and low-score `rho` —
were tuned **only** on 2023-24 (best: `xi=0.00125`/day ≈ 2-season half-life,
`rho=-0.10`); evaluation is on the untouched 2024-25 season. The team-strength
model is refit monthly during the test season, exactly as it would run live.
RPS = ranked probability score (the standard football metric; lower is
better). Code: `goal_models.py`.

**Test 2024-25, all leagues (1,730 matches):**

| Model | Accuracy | Log loss | RPS |
|---|---|---|---|
| Elo logistic regression (part-1 baseline) | 51.1% | 1.0006 | 0.2032 |
| Dixon-Coles Poisson (attack/defence, decay, rho) | 49.8% | 1.0003 | 0.2037 |
| Feature Poisson (lag-5 form + Elo) | 51.2% | 1.0014 | 0.2037 |
| **Blend: DC + feature Poisson + Elo-LR** | 51.0% | **0.9952** | **0.2018** |

Premier-League-only test (260 matches): blend reaches 52.3% / 0.9976 / 0.2082.

Takeaways:

- The blend is the **first model to clearly beat Elo alone on probability
  quality** (log loss 0.995 vs 1.001) — the three models are diverse enough
  that averaging helps, even though no single one dominates.
- Goal models give you the full score distribution for free
  (over/under, correct-score, expected points), which classifiers can't.
- The remaining gap to bookmaker closing odds (~0.95 log loss) is consistent
  with the information you don't have: lineups, injuries, market flow.
- Known limitation: newly promoted teams with no league history enter the DC
  model as league-average (all-zero dummies), which overrates them; using the
  bundled Championship data to initialize them is the obvious next step.

## 6. Is finishing skill or luck? (xG persistence study)

928 team-seasons across all 6 leagues, each split into first/second half
(code: `xg_analysis.py`, figure: `xg_analysis.png`):

| Quantity | Split-half correlation |
|---|---|
| xG created per game | **0.76** — strongly persistent (real skill) |
| Finishing over-performance (goals − xG) per game | **0.14** — barely persistent (mostly luck) |
| Finishing over-performance, season t vs t+1 (742 pairs) | 0.19 |

And the classic result reproduces: **first-half xG difference predicts
second-half points (r = 0.71) better than first-half goal difference
(r = 0.69) or first-half points themselves (r = 0.66).** A team's underlying
chance creation is more informative about its future than its actual results.

Practical consequences for modeling and analysis on this dataset:

- Feed models xG-based form rather than goals-based form; goals contain a
  large luck component that regresses within the same season.
- "Over-performing xG" is a sell signal, not a skill — useful for punditry
  claims, fantasy decisions, and value detection against naive league tables.
- The small but nonzero season-to-season persistence (~0.19) is consistent
  with elite finishers being real but rare — a nice follow-up study at the
  player level.

## Files in this folder

- `build_and_evaluate.py` — feature pipeline (rolling form + Elo) and
  classifier evaluation from the SQLite DB.
- `evaluate_library_csv.py` — evaluation of the library's own CSV export,
  including the leakage demonstration.
- `goal_models.py` — Dixon-Coles + feature Poisson + blends, probability
  evaluation (accuracy / log loss / RPS).
- `xg_analysis.py` — xG persistence and finishing-luck study; writes
  `xg_analysis.png`.
- `results_*.json` — all metrics and confusion matrices.

Reproduce: load `premier_league/data/premier_league.sql` into a SQLite file,
then `python research/build_and_evaluate.py --db <file>`.
