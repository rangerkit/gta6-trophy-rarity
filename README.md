# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rangerkit/gta6-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rangerkit/gta6-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the make.fun competition **gta6-trophy-rarity-forecast-01**: forecast, for every
PlayStation 5 trophy of Grand Theft Auto VI, the percent of tracked players who will have
earned it on Exophase one week after launch. `main.py` (Python 3.13, standard library only)
reads each case file in the input directory and writes five quantiles (5/25/50/75/95%) of
every trophy's earn rate at the case's target instant. Scoring is mean pinball loss on the
logit scale, so all modelling happens in logit space.

## Approach

- **Trend regime** — once the tracked-player pool has refreshed during a case's history,
  each trophy's logit rate is fitted by recency-weighted least squares against a saturating
  clock, ln(1 + hours since release). The fitted slope is shrunk toward zero and the total
  extrapolated move capped, so a steep launch-night slope is not run unchecked for days.
  The spread combines the house random walk (0.5 logit per √day), rescaled by the trophy's
  own observed step volatility, a term proportional to the extrapolated move, and the fit's
  residuals.
- **Pre-refresh regime** — if `players_tracked` never changed in the history, every read is
  one stale page and a refresh is still to come. The first refresh multiplies the pool
  several-fold while earners grow more slowly, so most rates step *down*: centres shift
  down in logit (least for high-rate trophies, whose earners keep pace) with a wide,
  roughly one-logit spread.
- **Boundary trophies** — a trophy still at 0.00 in every read gets a small positive centre
  with a wide upper tail (earners can only grow from zero); one pinned at 100.00 gets a
  centre just below the clamp with a wide lower tail (a growing pool always contains
  players who have not earned it yet).
- Degenerate histories (no reads, a single read, all-null rates) fall back to
  baseline-like behaviour. The program is a pure function of the case file — deterministic,
  byte-identical across runs, and far inside the 8-second limit.

## Running

```
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
