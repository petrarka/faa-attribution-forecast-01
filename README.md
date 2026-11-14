# faa-attribution-forecast-01

Offline quantile forecaster for the make.fun competition
**polymarket-faa-attribution-01**. For each market of each case it forecasts
the Yes price over the next `horizon_minutes` (240) minutes at five quantile
levels (0.05, 0.25, 0.5, 0.75, 0.95).

Python 3.10+, **standard library only**, no network. Deterministic: two runs
produce byte-identical output.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` (JSON case) from `$MAKEFUN_INPUT_DIR` (default `/input/`)
and writes `$MAKEFUN_OUTPUT_DIR/<id>.out` (default `/output/`) with
`{"forecasts":[{"id","q":[[q05,q25,q50,q75,q95] x 240]}]}`.

## Model

For each market with minute prices `hist` (last element = price at checkpoint):

- **Center** — `c = 0.3*hist[-1] + 0.7*median(hist[-22:])`: the median of the
  last 22 minutes blended with the last price.
- **Sigma** — `s = pstdev` of the last 60 one-minute deltas, clipped to
  `[0.0015, 0.02]`.
- **Quantiles** — `q_tau(h) = clamp01(c + z_tau * s * h**0.45)`, `h = 1..240`,
  with a leptokurtic z-vector: outer (5%/95%) quantiles stretched ×1.3, inner
  (25%/75%) tightened ×0.9. Rounded to 6 decimals.

Short histories are handled gracefully (sigma falls back to the 0.0015 floor
when there is no delta).

## Practice results (`check.py`)

Pseudo-checkpoints (t = 720..len-240 step 60) on the practice kit p00–p05,
pinball loss vs a persistence baseline (last price, sigma 0.0025, sqrt(h)):

| model | pinball | vs baseline |
|-------|---------|-------------|
| v1 (backtest frame) | 0.017980 | −11.9% |
| v2 (+ outer stretch) | 0.017903 | −12.2% |
| v3 (+ inner tighten) | **0.017839** | **−12.6%** |

Run the checks:

```sh
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 check.py
```

## Tried and rejected (backtest/practice)

- Long-window median / logit transform / mean reversion to a long median — hurt.
- `h**0.5` (random-walk spread) — worse than `h**0.45`.
- MAD-based volatility and longer (90/120 min) volatility windows — worse.
- Global sigma multiplier, median-window (18–26) and blend-weight sweeps — the
  backtest frame is already at the optimum; only the leptokurtic shape gave a
  consistent gain, on every practice case.

## Files

- `main.py` — the forecaster.
- `check.py` — format validator + pseudo-checkpoint pinball scoring vs baseline.