"""Forecast GTA VI trophy earn rates on Exophase one week after launch.

Reads every case file in the input directory (one JSON object per case:
the game's 55 trophies and the house's reads of Exophase's trophy page
before the case's checkpoint) and writes, for each case, five quantiles
(5/25/50/75/95%) of each trophy's earn rate at the case's target.

Model, per trophy, on the logit scale:
  * Fit a least-squares line of logit(rate) against ln(1 + hours since
    2026-11-19T00:00Z). Consecutive reads that repeat the previous rate
    are dropped first, so a frozen page cannot weigh one level twice.
  * Shrink each slope 25% toward the median slope across all trophies,
    and damp the extrapolation toward the target.
  * If the whole history is perfectly flat over at least an hour, the
    page is treated as frozen and due a refresh: the median logit is
    shifted down by 1.2 and extra variance is added, because a refresh
    dilutes every rate as newly tracked players join with fewer trophies.
  * The spread combines a random-walk term over the horizon, the fit's
    residual scatter, and slope uncertainty, widened further by the
    refresh prior when it applies.
  * Medians are made non-increasing along the story chain
    (chapters 1-4, then the finale, the 100% trophy, then platinum)
    and the platinum is capped at the lowest other trophy's median.
  * Quantiles are the normal quantiles of the median logit plus/minus
    the spread, mapped back to percent, rounded half-up to 4 decimals.

Standard library only. Deterministic: same input bytes give the same
output bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6449, -0.6745, 0.0, 0.6745, 1.6449]

# Launch day begins here; the log-time axis is measured from this instant.
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

RATE_LOW = 0.01
RATE_HIGH = 99.99

# Random-walk spread, in logit units per square-root day.
WALK_SIGMA = 0.5
# How far the median logit falls when a frozen page is due a refresh.
REFRESH_SHIFT = 1.2
# Extra logit spread (standard deviation) that comes with a refresh.
REFRESH_SPREAD = 0.6
# Slope shrinkage toward the cross-trophy median slope.
SHRINK = 0.25
# Extrapolation damping per unit of log-time beyond the last read.
DAMP_RATE = 0.15
# The story chain, in order: chapters 1-4, finale, 100%, platinum.
CHAIN = [3, 4, 5, 6, 7, 14, 1]


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log_hours(moment):
    """ln(1 + hours from the epoch to `moment`)."""
    hours = (moment - EPOCH).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


def clamp_rate(pct):
    return min(RATE_HIGH, max(RATE_LOW, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(value):
    return 100.0 / (1.0 + math.exp(-value))


def round4(value):
    """Round to 4 decimals, half up, as JS Math.round(x * 10000) / 10000.

    Whole numbers come back as ints so the JSON writes 100, not 100.0.
    """
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 10000.0
    return int(rounded) if rounded.is_integer() else rounded


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def fit_line(xs, ys):
    """Least-squares line y = a + b*x, returned as (y_bar, x_bar, b, resid).

    With too few distinct x values, or no variation in y, the slope is 0
    and the line is the flat mean. resid is the population standard
    deviation of the residuals, floored later by the caller.
    """
    n = len(xs)
    x_bar = sum(xs) / n
    y_bar = sum(ys) / n
    sxx = sum((x - x_bar) ** 2 for x in xs)
    if n < 3 or len(set(xs)) < 3 or sxx <= 1e-12:
        b = 0.0
    else:
        sxy = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
        b = sxy / sxx
    residuals = [y - (y_bar + b * (x - x_bar)) for x, y in zip(xs, ys)]
    var = sum(r * r for r in residuals) / n
    resid = math.sqrt(var)
    return y_bar, x_bar, b, resid, sxx, n


def pool_adjacent_violators(values):
    """Make `values` non-increasing by averaging adjacent violators.

    Classic PAVA: scan left to right, and whenever a block's average is
    below the next block's, merge the two blocks and re-check.
    """
    blocks = [[v, 1] for v in values]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0] - 1e-12:
            total = blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]
            count = blocks[i][1] + blocks[i + 1][1]
            blocks[i:i + 2] = [[total / count, count]]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = []
    for avg, count in blocks:
        out.extend([avg] * count)
    return out


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")

    trophies = case["trophies"]
    n_trophies = len(trophies)
    target = parse_time(case["target"])
    x_target = log_hours(target)
    horizon_days = case["horizon_minutes"] / 1440.0

    slots = [parse_time(read["slot"]) for read in history]
    x_reads = [log_hours(slot) for slot in slots]
    x_last = x_reads[-1]

    # A frozen page: every read reports exactly the same rates, and the
    # reads span at least an hour. Only then does the refresh prior fire.
    all_rates = [read["rates"] for read in history]
    history_flat = all(rates == all_rates[0] for rates in all_rates)
    span_minutes = (slots[-1] - slots[0]).total_seconds() / 60.0
    refresh_due = history_flat and span_minutes >= 60.0

    # Fit one line per trophy on its de-duplicated points.
    fits = []
    for idx in range(n_trophies):
        xs, ys = [], []
        prev_rate = None
        for read, x in zip(history, x_reads):
            rate = read["rates"][idx]
            if rate is None:
                continue
            if prev_rate is not None and rate == prev_rate:
                continue  # frozen read: same level as the one before it
            prev_rate = rate
            xs.append(x)
            ys.append(logit(rate))
        if not xs:
            xs = [x_last]
            ys = [logit(RATE_LOW)]
        fits.append(fit_line(xs, ys))

    # Shrink every slope toward the median slope across trophies.
    slope_med = median([fit[2] for fit in fits])

    medians = {}
    spreads = {}
    dx = x_target - x_last
    dx_pos = max(0.0, dx)
    damp = 1.0 / (1.0 + DAMP_RATE * dx_pos)
    for idx, trophy in enumerate(trophies):
        y_bar, x_bar, b, resid, sxx, n_points = fits[idx]
        b_shrunk = (1.0 - SHRINK) * b + SHRINK * slope_med
        m = y_bar + b_shrunk * damp * (x_target - x_bar)
        # With fewer than 3 points the fit carries no residual signal,
        # so fall back to a fixed scatter; otherwise clamp the residual.
        if n_points < 3:
            resid = 0.15
        else:
            resid = min(1.0, max(0.05, resid))
        se_b = resid / math.sqrt(sxx) if sxx > 1e-12 else 0.0
        var = (WALK_SIGMA * math.sqrt(horizon_days)) ** 2
        var += resid ** 2
        var += (se_b * dx_pos) ** 2
        if refresh_due:
            m -= REFRESH_SHIFT
            var += REFRESH_SPREAD ** 2
        s = math.sqrt(var)
        s = min(2.0, max(0.30, s))
        medians[trophy["id"]] = m
        spreads[trophy["id"]] = s

    # Story-chain monotonicity, in logit space, via PAVA.
    chain_values = [medians[tid] for tid in CHAIN if tid in medians]
    chain_fixed = pool_adjacent_violators(chain_values)
    for tid, value in zip([t for t in CHAIN if t in medians], chain_fixed):
        medians[tid] = value

    # The platinum can never sit above the easiest other trophy.
    if 1 in medians:
        others = [m for tid, m in medians.items() if tid != 1]
        if others:
            medians[1] = min(medians[1], min(others))

    rows = []
    for trophy in trophies:
        tid = trophy["id"]
        m = medians[tid]
        s = spreads[tid]
        qs = []
        for z in Z:
            q = round4(inv_logit(m + z * s))
            qs.append(min(100, max(0, q)))
        # Force non-decreasing across the row.
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
