"""gta6-trophy-rarity-forecast-01 entry: a per-trophy logit-space forecaster.

Written for the make.fun competition "gta6-trophy-rarity-forecast-01". The program is run once
per case (`python3 -P -s /bot/main.py`, Python 3.13, standard library only, no network): it
reads every <id>.in in the input directory and writes <id>.out to the output directory, one row
per trophy holding five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of that trophy's Exophase earn
rate, in percent, at the case's target instant.

How it forecasts, per trophy (the score is mean pinball loss on the logit scale, so all of the
work happens in logit space, with rates clamped to 0.01..99.99 exactly as the scorer clamps):

  * Trend regime (the pool has refreshed during the history): fit a recency-weighted least
    squares line of logit(rate) against ln(1 + hours since 2026-11-19T00:00Z) -- a saturating
    clock, since earn rates grow fastest in the first hours. The fitted slope is shrunk toward
    zero before it is extrapolated to the target, and the total move is capped: a launch-week
    slope never runs unchecked for days. The spread combines the house random walk
    (0.5 logit per sqrt-day), rescaled by the trophy's own observed step volatility, a term
    proportional to the extrapolated move, and the fit's residuals.

  * Pre-refresh regime (players_tracked never changed in the whole history, so every read is
    the same stale page and a refresh is still to come): the first refresh multiplies the
    tracked pool several-fold while earners grow more slowly, so most rates step DOWN. The
    centre is shifted down in logit (less for high-rate trophies, whose earners keep pace
    with the pool) and the spread is set wide, near a full logit unit.

  * Boundary trophies: one still at 0.00 in every read gets a small positive centre with a
    wide upper tail (its earners can only grow from zero); one pinned at 100.00 gets a centre
    a little below the clamp with a wide lower tail (a growing pool always contains players
    who have not earned it yet).

Degenerate inputs (no reads, a single read, all-null rates) fall back to house-baseline
behaviour: last rate held flat with the random-walk spread. Everything is a pure function of
the case file -- no randomness, no clock, no state -- so two runs write identical bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input
and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, in case a helper module is
# ever shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z_TABLE = {-1.645: 0.05}
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The spread of a rate's move, in logit units per square-root day (the house baseline's).
RW_SIGMA = 0.5
# The saturating clock's origin: midnight UTC on release day, as the house curve baseline uses.
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Model constants (see the module docstring for the reasoning).
SLOPE_SHRINK = 0.8        # how much of a fitted log-time slope is extrapolated
MAX_MOVE = 3.0            # cap on the extrapolated move, in logit units
WEIGHT_HALF_LIFE_H = 20.0  # recency weighting of reads in the trend fit
FLAT_SHIFT = 0.8          # pre-refresh dilution shift at a 0% rate, in logit units
FLAT_SIGMA = 1.05         # pre-refresh spread, in logit units
ZERO_CENTER_PCT = 0.05    # centre for a trophy still at 0.00 in every read
ZERO_SIGMA = 1.15
FULL_CENTER_PCT = 98.2    # centre for a trophy pinned at 100.00
FULL_SIGMA = 0.8


def clamp_rate(x: float) -> float:
    """Clamp a percent rate to the scorer's 0.01..99.99."""
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def norm_ppf(p: float) -> float:
    """Standard normal quantile (Acklam's approximation), for non-standard level sets."""
    if p in (0.05, 0.25, 0.5, 0.75, 0.95):
        return Z[LEVELS.index(p)]
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 does.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100.
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_epoch(when: datetime) -> float:
    return (when - EPOCH).total_seconds() / 3600.0


def log_clock(hours: float) -> float:
    """The saturating clock: ln(1 + hours since release midnight), floored at 0."""
    return math.log1p(max(hours, 0.0))


def make_row(trophy_id: int, center_logit: float, sigma: float, zs: list[float]) -> dict:
    qs = []
    for z in zs:
        q = inv_logit(center_logit + z * sigma)
        qs.append(round4(min(100.0, max(0.0, q))))
    # Rounding can only flatten the order, never break it, but enforce monotonicity outright.
    for k in range(1, len(qs)):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    return {"id": trophy_id, "q": qs}


def weighted_line(xs: list[float], ys: list[float], ws: list[float]):
    """Weighted least squares y ~ a + b*x; returns (a, b, weighted residual RMS)."""
    sw = sum(ws)
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw)
    return a, b, resid


def step_volatility(series: list[tuple[float, float]]) -> float:
    """RMS step size of logit(rate) per sqrt-day, over consecutive distinct states.

    `series` is (hours, rate) pairs in read order. A coarse refresh clock makes this an
    underestimate of the true diffusion, so callers treat it as a rescaling signal only.
    """
    moves = []
    prev_h, prev_y = series[0][0], logit(series[0][1])
    for h, rate in series[1:]:
        y = logit(rate)
        if y != prev_y:
            dt_days = max((h - prev_h) / 24.0, 1e-6)
            moves.append(abs(y - prev_y) / math.sqrt(dt_days))
            prev_h, prev_y = h, y
    if not moves:
        return 0.0
    return math.sqrt(sum(m * m for m in moves) / len(moves))


def forecast_trophy(series: list[tuple[float, float]], horizon_days: float,
                    x_target: float, pool_never_refreshed: bool, zs: list[float],
                    trophy_id: int) -> dict:
    """One trophy's row. `series` holds (hours since epoch, rate) for its non-null reads."""
    base_sigma = RW_SIGMA * math.sqrt(max(horizon_days, 0.0))
    if not series:
        # No observation at all: the baseline's answer for an unknown rate is the clamp floor.
        return make_row(trophy_id, logit(0.0), max(base_sigma, 0.3), zs)

    last_rate = series[-1][1]
    last_logit = logit(last_rate)

    if pool_never_refreshed:
        # Every read so far is one stale page; a refresh lands before the target.
        if last_rate <= 0.0:
            return make_row(trophy_id, logit(0.03), ZERO_SIGMA, zs)
        if last_rate >= 100.0:
            return make_row(trophy_id, logit(97.5), FULL_SIGMA, zs)
        shift = FLAT_SHIFT - 0.5 * (last_rate / 100.0)
        return make_row(trophy_id, last_logit - shift, FLAT_SIGMA, zs)

    if last_rate <= 0.0 and all(rate <= 0.0 for _, rate in series):
        # Never earned by anyone tracked, in a pool that has refreshed: earners can only
        # grow from zero, so centre small and positive with a wide upper tail.
        return make_row(trophy_id, logit(ZERO_CENTER_PCT), ZERO_SIGMA, zs)
    if last_rate >= 100.0 and all(rate >= 100.0 for _, rate in series):
        # Earned by everyone tracked so far; a growing pool dilutes it a little.
        return make_row(trophy_id, logit(FULL_CENTER_PCT), FULL_SIGMA, zs)

    distinct = {rate for _, rate in series}
    if len(distinct) == 1 or len(series) < 3:
        # A rate that never moved while the pool refreshed around it is in equilibrium.
        return make_row(trophy_id, last_logit, max(base_sigma * 0.9, 0.3), zs)

    # Trend regime: recency-weighted fit of logit(rate) on the saturating clock.
    xs = [log_clock(h) for h, _ in series]
    ys = [logit(rate) for _, rate in series]
    last_h = series[-1][0]
    ws = [0.5 ** (max(last_h - h, 0.0) / WEIGHT_HALF_LIFE_H) for h, _ in series]
    _, slope, resid = weighted_line(xs, ys, ws)
    slope = min(6.0, max(-6.0, slope))
    dx = max(x_target - xs[-1], 0.0)
    move = SLOPE_SHRINK * slope * dx
    move = min(MAX_MOVE, max(-MAX_MOVE, move))
    center = last_logit + move

    vol = step_volatility(series)
    vol_ratio = min(1.8, max(0.7, vol / RW_SIGMA)) if vol > 0 else 0.85
    sigma = math.sqrt((RW_SIGMA * vol_ratio) ** 2 * max(horizon_days, 0.0)
                      + (0.30 * abs(move)) ** 2
                      + (min(resid, 1.0) * 0.5 * min(dx, 2.0)) ** 2
                      + 0.02)
    sigma = min(2.4, max(0.3, sigma))
    return make_row(trophy_id, center, sigma, zs)


def forecast_case(case: dict) -> dict:
    levels = case.get("levels") or LEVELS
    zs = [norm_ppf(p) for p in levels]
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = (case.get("horizon_minutes") or 0) / 1440.0
    x_target = log_clock(hours_since_epoch(parse_time(case["target"])))

    pool_never_refreshed = len({read.get("players_tracked") for read in history}) <= 1

    rows = []
    for i, trophy in enumerate(trophies):
        series = []
        for read in history:
            rates = read.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            if rate is not None:
                series.append((hours_since_epoch(parse_time(read["at"])), float(rate)))
        rows.append(forecast_trophy(series, horizon_days, x_target,
                                    pool_never_refreshed, zs, trophy["id"]))
    return {"trophies": rows}


def baseline_case(case: dict) -> dict:
    """The house baseline, used only if the model itself trips on a malformed case."""
    last = (case.get("history") or [{}])[-1]
    rates = last.get("rates") or []
    spread = RW_SIGMA * math.sqrt(max((case.get("horizon_minutes") or 0) / 1440.0, 0.0))
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = rates[i] if i < len(rates) else None
        lg = logit(rate if rate is not None else 0.0)
        rows.append(make_row(trophy["id"], lg, spread, Z))
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = baseline_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
