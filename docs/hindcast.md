# Hindcast benchmark

After creating a case directory, `uv run stormfork benchmark <case-directory>` reads
its `case.json`, writes `benchmark.json` beside it and prints the path. A case
names its IBTrACS CSV, its a-deck and one matched track table per init. Late
model runs and forecaster-facing products stay in separate groups, each with the
homogeneous sample size `n` and skill against OCD5.

A minimal case is:

```json
{
  "id": "hindcast_milton_2024",
  "storm": "AL142024",
  "init_source": "era5",
  "ibtracs": "ibtracs.csv",
  "adeck": "aal142024.dat.gz",
  "runs": [{"init": "2024-10-05T00:00:00", "tracks": "tracks-2024100500.csv"}]
}
```

Omit `adeck` for a current-year NHC storm to read the public a-deck. Outside the
Atlantic, eastern Pacific and central Pacific the output says the benchmark does
not apply.

**Result:** none yet. The Milton hindcast tracks are not committed, so I publish
no score and make no forecast-skill claim.
A small hindcast is a case study, not a skill estimate; operational ensemble
means may also use different member counts.
