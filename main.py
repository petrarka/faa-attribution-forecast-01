#!/usr/bin/env python3
"""Offline quantile forecaster for polymarket-faa-attribution-01.

Reads each ``<id>.in`` case from ``$MAKEFUN_INPUT_DIR`` (default ``/input/``)
and writes ``$MAKEFUN_OUTPUT_DIR/<id>.out`` (default ``/output/``) with 5
quantiles per minute over the case horizon for every market.

Python 3.10+, standard library only, no network. Deterministic.
"""

from __future__ import annotations

import json
import math
import os
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

# Quantile levels the contest scores, in the order each row lists them.
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantile at each level, to 6 decimals.
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

# Price moves are leptokurtic: fatter tails and a more peaked center than a
# normal. Stretch the outer (5% / 95%) quantiles outward; the inner ones are
# tightened in v3. Monotone in every row because every factor is positive.
TAIL_SCALE = 1.3
Z_LEV = [z * TAIL_SCALE if abs(z) > 1.0 else z for z in Z]


def price6(x: float) -> float:
    """Clamp to [0, 1], round to 6 decimals (never -0.0)."""
    v = math.floor(max(0.0, min(1.0, x)) * 1e6 + 0.5) / 1e6
    return 0.0 if v == 0.0 else v


def median(xs: list[float]) -> float:
    """Median of a non-empty list (mean of the two middles for even counts)."""
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def center(prices: list[float]) -> float:
    """Robust blended center: median of the last 22 minutes blended with the last price."""
    last = prices[-1]
    window = prices[-22:]
    med = median(window)
    return 0.3 * last + 0.7 * med


def sigma(prices: list[float]) -> float:
    """Short-window volatility: pstdev of the last 60 one-minute deltas, clipped."""
    tail = prices[-61:] if len(prices) > 1 else prices[:]
    deltas = [tail[i] - tail[i - 1] for i in range(1, len(tail))]
    if not deltas:
        return 0.0015
    m = sum(deltas) / len(deltas)
    var = sum((d - m) ** 2 for d in deltas) / len(deltas)
    s = math.sqrt(var)
    return min(0.02, max(0.0015, s))


def forecast(prices: list[float], horizon: int) -> list[list[float]]:
    """Return horizon rows of five quantiles for one market."""
    if not prices:
        raise ValueError("market has no prices")
    c = center(prices)
    s = sigma(prices)
    rows: list[list[float]] = []
    for h in range(1, horizon + 1):
        spread = s * (h ** 0.45)
        rows.append([price6(c + z * spread) for z in Z_LEV])
    return rows


def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in names:
        case_id = name[: -len(".in")]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        horizon = int(case["horizon_minutes"])
        forecasts = [
            {"id": m["id"], "q": forecast(list(m["prices"]), horizon)}
            for m in case["markets"]
        ]
        out = os.path.join(OUTPUT_DIR, f"{case_id}.out")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"forecasts": forecasts}, separators=(",", ":")))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())