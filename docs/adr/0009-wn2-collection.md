# 0009: Store whole WN2 runs with member 0

Status: accepted.

## Context

The scoreboard needs historical forecasts identified by initialization, retaining every six-hour lead through 168 hours. An ensemble mean and a single member answer different comparison questions, so both must be stored.

## Decision

Read Open-Meteo's WN2 metadata on the Ensemble API host. Request all members for the exact advertised UTC horizon. Require the full member set, valid-time grid and documented units. Compute the mean over all members and retain the unsuffixed member 0. Write one parquet file with the six archive columns only after every town validates.

Run at 06:30 and 18:30 UTC. The workflow stores output on the data branch and serializes collector jobs. It creates that branch when first activated. Scheduled collection and GitHub acceptance are pending activation.

## Evidence and limits

On 2026-11-05, the advertised WN2 current run returned “not available” when requested with `run=2026-11-05T00:00`. The exact `start_hour`/`end_hour` window returned all members and the full horizon; the recorded request is in `tests/fixtures/wn2_source.json`. The metadata host must be `ensemble-api.open-meteo.com`; the same path on `api.open-meteo.com` returned 404.

The live endpoint does not provide a per-value initialization stamp. The collector waits ten minutes after advertised availability, follows [Open-Meteo's replication guidance](https://open-meteo.com/en/docs/model-updates), and checks initialization again after collection. This detects cycle changes but cannot independently prove that every replica served the advertised run. Named-run WN2 support should be rechecked before relying on the archive for archive scoring.

Scoring, keepalive and statistical decisions are separate from collection.

The `temporal_resolution=hourly_6` probe returned 29 timestamps but only 28 values per member for the requested horizon. That unchanged response is retained as `tests/fixtures/wn2_inconsistent_six_hourly.json`. The collector requests hourly output, validates all 169 timestamps and member values, then keeps the six-hourly points through 168 h. It never truncates a mismatched response to make it fit.
