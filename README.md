# faa-attribution-forecast-01

Offline quantile forecaster for make.fun competition polymarket-faa-attribution-01.

`main.py` (Python 3, stdlib only, no network) reads each `*.in` case from
`$MAKEFUN_INPUT_DIR` (default `/input/`) — minute Yes-prices of every Polymarket
market of the case up to a checkpoint — and writes
`$MAKEFUN_OUTPUT_DIR/<id>.out` (default `/output/`):
`{"forecasts":[{"id":"<market>","q":[[q05,q25,q50,q75,q95] x 240]}]}`.
Deterministic, well under the 8 s / 1 CPU / 1024 MB sandbox limit.

Model: robust blended center (median of last 22 minutes + last price) with
short-window volatility (60 one-minute deltas), quantiles spread as
z * sigma * h^0.45, clamped to [0,1], rounded to 6 decimals.
