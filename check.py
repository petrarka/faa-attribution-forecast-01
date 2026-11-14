#!/usr/bin/env python3
"""Practice checks for the polymarket-faa-attribution-01 forecaster.

Two things:

1. Format validation of every ``<id>.out`` produced by ``main.py``: exact keys,
   exactly one forecast per case market, exactly ``horizon`` rows of five finite
   numbers in [0, 1], non-decreasing, rounded to 6 decimals, file <= 1 MiB.

2. Pseudo-checkpoint scoring: for each market, walk ``t`` from 720 to len-240 in
   steps of 60, forecast from ``prices[:t]`` and score the five quantiles
   against the true ``prices[t:t+240]`` with the pinball loss. Compare the model
   to the persistence baseline (center = last price, sigma 0.0025, exponent 0.5).

Usage:
    MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 check.py
"""

from __future__ import annotations

import json
import os
import sys

import main as M

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

QUANTILES = M.QUANTILES
Z = M.Z
TAU = QUANTILES
START_T = 720
STEP = 60
HORIZON = 240


def pinball(q: float, y: float, tau: float) -> float:
    """Standard quantile (pinball) loss for one prediction at level tau."""
    return (y - q) * tau if y >= q else (q - y) * (1.0 - tau)


def validate_outputs() -> int:
    """Return the number of validated case files; raises on any format problem."""
    problems: list[str] = []
    names = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    count = 0
    for name in names:
        case_id = name[: -len(".in")]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        horizon = int(case["horizon_minutes"])
        want = [m["id"] for m in case["markets"]]
        path = os.path.join(OUTPUT_DIR, f"{case_id}.out")
        if not os.path.exists(path):
            problems.append(f"{case_id}: missing output file")
            continue
        size = os.path.getsize(path)
        if size > 1024 * 1024:
            problems.append(f"{case_id}: output too large ({size} bytes)")
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if set(data.keys()) != {"forecasts"}:
            problems.append(f"{case_id}: bad top-level keys {sorted(data.keys())}")
            continue
        fc = data["forecasts"]
        got = [f["id"] for f in fc]
        if got != want:
            problems.append(f"{case_id}: forecast ids differ from case markets")
        for f in fc:
            if set(f.keys()) != {"id", "q"}:
                problems.append(f"{case_id}/{f.get('id')}: bad keys {sorted(f.keys())}")
            q = f["q"]
            if len(q) != horizon:
                problems.append(f"{case_id}/{f['id']}: {len(q)} rows != {horizon}")
            for hi, row in enumerate(q):
                if len(row) != 5:
                    problems.append(f"{case_id}/{f['id']}[{hi}]: {len(row)} cols")
                    continue
                for j, v in enumerate(row):
                    if not isinstance(v, (int, float)) or isinstance(v, bool):
                        problems.append(f"{case_id}/{f['id']}[{hi}][{j}]: not a number")
                        continue
                    if not (0.0 <= v <= 1.0):
                        problems.append(f"{case_id}/{f['id']}[{hi}][{j}]: {v} out of [0,1]")
                    if round(v, 6) != v:
                        problems.append(f"{case_id}/{f['id']}[{hi}][{j}]: {v} not 6dp")
                for j in range(4):
                    if row[j] > row[j + 1] + 1e-12:
                        problems.append(
                            f"{case_id}/{f['id']}[{hi}]: row not non-decreasing {row}"
                        )
                        break
        count += 1

    if problems:
        print("FORMAT PROBLEMS:")
        for p in problems[:50]:
            print("  " + p)
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more")
        return -1
    print(f"FORMAT OK: {count} case file(s), all valid.")
    return count


def baseline_rows(prices: list[float], horizon: int) -> list[list[float]]:
    """Persistence baseline: center = last price, sigma 0.0025, sqrt(h)."""
    x = prices[-1]
    rows = []
    for h in range(1, horizon + 1):
        spread = 0.0025 * (h ** 0.5)
        rows.append([M.price6(x + z * spread) for z in Z])
    return rows


def score_model() -> tuple[float, float]:
    """Return (model pinball, baseline pinball) averaged over pseudo-checkpoints."""
    names = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    sum_model = 0.0
    sum_base = 0.0
    n = 0
    for name in names:
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        horizon = int(case["horizon_minutes"])
        for m in case["markets"]:
            prices = [float(v) for v in m["prices"]]
            length = len(prices)
            t = START_T
            while t <= length - horizon:
                hist = prices[:t]
                truth = prices[t : t + horizon]
                m_rows = M.forecast(hist, horizon)
                b_rows = baseline_rows(hist, horizon)
                for hi in range(horizon):
                    y = truth[hi]
                    for j, tau in enumerate(TAU):
                        sum_model += pinball(m_rows[hi][j], y, tau)
                        sum_base += pinball(b_rows[hi][j], y, tau)
                        n += 1
                t += STEP
    if n == 0:
        print("No pseudo-checkpoints available (markets too short?).")
        return (float("nan"), float("nan"))
    return (sum_model / n, sum_base / n)


def main() -> int:
    if validate_outputs() < 0:
        return 1
    model, base = score_model()
    print(f"Pinball model    : {model:.6f}")
    print(f"Pinball baseline : {base:.6f}")
    if base > 0:
        print(f"Delta            : {model - base:+.6f} "
              f"({(model - base) / base * 100:+.2f}% vs baseline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())