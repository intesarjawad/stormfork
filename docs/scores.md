# The scoreboard: collecting and scoring public forecasts

The collector writes whole WeatherNext 2 runs for coastal stations:

```sh
uv run stormfork scores collect --output /tmp/stormfork-data
```

Collection exits nonzero if upstream metadata is stale, future-dated or still
replicating. It leaves stored runs untouched. For stale metadata, retry the same
command in 30 minutes. Check the advertised initialization and earliest safe
retry time with:

```sh
uv run python - <<'PYCODE'
from datetime import UTC, datetime
from stormfork.scores.fetch import METADATA_URL, REPLICATION_DELAY, RunMetadata, request_bytes
metadata = RunMetadata.model_validate_json(request_bytes(METADATA_URL))
print("Now:", datetime.now(UTC).isoformat())
print("Initialization:", metadata.init_time.isoformat())
available = datetime.fromtimestamp(metadata.last_run_availability_time, UTC)
print("Replicated after:", (available + REPLICATION_DELAY).isoformat())
PYCODE
```

A retry still fails until Open-Meteo advertises a current run and its replication
window has passed. Do not change the metadata or substitute an older run as a
successful collection.

Scoring reads local files and can run even when collection fails. Fetch the
observations and IFS inputs separately when the upstream sources are available:

```sh
uv run stormfork scores fetch --output /tmp/stormfork-data/score-inputs
```

Build from the last complete stored inputs without fetching anything:

```sh
uv run stormfork scores build \
  --wn2 /tmp/stormfork-data/wn2 \
  --scoring-day "$(cat /tmp/stormfork-data/score-inputs/scoring-day.txt)" \
  --observations /tmp/stormfork-data/score-inputs/observations.csv \
  --ifs /tmp/stormfork-data/score-inputs/ifs.csv \
  --output site/public/scores/scores.json
```

Missing or invalid inputs fail the build. The board retains source dates and its
staleness warning. The fetch command records the scoring day beside its CSVs;
rebuilding stored inputs uses that date, so it does not make old inputs appear
current. The scheduled workflow stores complete scoring inputs on the
data branch and can rebuild and deploy from them after a failed fresh fetch;
the collection failure remains visible.

The scheduled workflow needs activation before collection starts. See [publication notes](publication.md).
