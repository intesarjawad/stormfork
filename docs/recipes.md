# Recipes

A recipe is a YAML file that describes one experiment. The files in `experiments/` are recipes, and a recipe of your own has the same format. [Method](method.md) says what the three runs of an experiment are and what is compared.

## Schema

The schema below is printed from the model that loads every recipe, `stormfork.scenarios.Scenario`. `uv run python scripts/recipes_schema.py` prints it again, and `uv run python scripts/recipes_schema.py --check docs/recipes.md` fails if this copy has drifted from the model.

<!-- recipe-schema -->
```json
{
  "$defs": {
    "JsonValue": {},
    "PerturbationInfo": {
      "additionalProperties": false,
      "description": "Edit type and its JSON-serializable physical parameters.",
      "properties": {
        "type": {
          "minLength": 1,
          "pattern": "\\S",
          "title": "Type",
          "type": "string"
        },
        "params": {
          "additionalProperties": {
            "$ref": "#/$defs/JsonValue"
          },
          "title": "Params",
          "type": "object"
        }
      },
      "required": [
        "type",
        "params"
      ],
      "title": "PerturbationInfo",
      "type": "object"
    },
    "Seeds": {
      "additionalProperties": false,
      "description": "The fork shares baseline seeds; control uses baseline plus 1000.",
      "properties": {
        "baseline": {
          "minimum": 0,
          "title": "Baseline",
          "type": "integer"
        },
        "control": {
          "minimum": 0,
          "title": "Control",
          "type": "integer"
        }
      },
      "required": [
        "baseline",
        "control"
      ],
      "title": "Seeds",
      "type": "object"
    },
    "Sweep": {
      "additionalProperties": false,
      "description": "One edit parameter taken at several values, one run per value.",
      "properties": {
        "param": {
          "minLength": 1,
          "pattern": "\\S",
          "title": "Param",
          "type": "string"
        },
        "values": {
          "items": {
            "type": "number"
          },
          "minItems": 2,
          "title": "Values",
          "type": "array"
        }
      },
      "required": [
        "param",
        "values"
      ],
      "title": "Sweep",
      "type": "object"
    },
    "TaperPlacement": {
      "additionalProperties": false,
      "description": "Where an edit's taper sits at the experiment's init time.",
      "properties": {
        "centre": {
          "maxItems": 2,
          "minItems": 2,
          "prefixItems": [
            {
              "maximum": 90,
              "minimum": -90,
              "type": "number"
            },
            {
              "maximum": 360,
              "minimum": -180,
              "type": "number"
            }
          ],
          "title": "Centre",
          "type": "array"
        },
        "radius_km": {
          "exclusiveMinimum": 0,
          "title": "Radius Km",
          "type": "number"
        }
      },
      "required": [
        "centre",
        "radius_km"
      ],
      "title": "TaperPlacement",
      "type": "object"
    }
  },
  "additionalProperties": false,
  "description": "One experiment; a hindcast has several inits and no edit.\n\nThe title and limits are the text the manifest carries. A published experiment\nneeds both; a recipe someone runs for themselves may leave them out.",
  "properties": {
    "id": {
      "pattern": "^[a-z0-9][a-z0-9_]*$",
      "title": "Id",
      "type": "string"
    },
    "title": {
      "anyOf": [
        {
          "minLength": 1,
          "pattern": "\\S",
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Title"
    },
    "kind": {
      "enum": [
        "fork",
        "sweep",
        "hindcast"
      ],
      "title": "Kind",
      "type": "string"
    },
    "storm": {
      "minLength": 1,
      "pattern": "\\S",
      "title": "Storm",
      "type": "string"
    },
    "init": {
      "items": {
        "format": "date-time",
        "type": "string"
      },
      "minItems": 1,
      "title": "Init",
      "type": "array"
    },
    "source": {
      "enum": [
        "era5",
        "hres_sample",
        "ifs_open_hybrid"
      ],
      "title": "Source",
      "type": "string"
    },
    "edit": {
      "anyOf": [
        {
          "$ref": "#/$defs/PerturbationInfo"
        },
        {
          "type": "null"
        }
      ],
      "default": null
    },
    "sweep": {
      "anyOf": [
        {
          "$ref": "#/$defs/Sweep"
        },
        {
          "type": "null"
        }
      ],
      "default": null
    },
    "taper": {
      "anyOf": [
        {
          "$ref": "#/$defs/TaperPlacement"
        },
        {
          "type": "null"
        }
      ],
      "default": null
    },
    "seeds": {
      "$ref": "#/$defs/Seeds"
    },
    "limits": {
      "items": {
        "minLength": 1,
        "pattern": "\\S",
        "type": "string"
      },
      "title": "Limits",
      "type": "array"
    }
  },
  "required": [
    "id",
    "kind",
    "storm",
    "init",
    "source",
    "seeds"
  ],
  "title": "Scenario",
  "type": "object"
}
```

The schema fixes the shape of a recipe, and the loader adds the checks it cannot express:

- `kind` is `fork`, `sweep` or `hindcast`. A fork or sweep has one init and one edit. A hindcast has one or more inits and no edit, sweep or taper.
- `init` times are UTC and fall on the six-hour cycle: hours 00, 06, 12 and 18.
- `relocate` and `scale_intensity` need a `taper`. `sst_anomaly` and `noise` must not have one. Take a taper from the storm's best-track row with `stormfork scenarios taper <storm> --ibtracs <csv> --init <time>`.
- A sweep names one parameter in `sweep` and leaves it out of `edit.params`. Its values are distinct and include the one that leaves the input unedited, the null fork: 0 for `east_km`, `north_km`, `delta_k` and `amplitude`, 1 for `factor`. `sigma_km` and `seed` have no such value, so they cannot be swept.
- `seeds.control` is `seeds.baseline` plus 1000.
- `title` and `limits` are optional for a recipe you run yourself. A published experiment needs both. A storm before 2024 also carries the sentence that it is inside the model's training period; `tests/test_limits.py` enforces that for the published list.

## One worked recipe per edit type

All five start from the HRES sample on 2024-10-07 at 00 UTC, because `rerun` fetches only that source. Each loads with the scenario loader, which `tests/test_recipes_doc.py` checks. The storm is Milton and the tapers are the published ones for that time.

Move the starting position:

```yaml
id: my_relocation
title: Hurricane Milton, starting position moved 100 km west
kind: fork
storm: 2024279N21265
init: 2024-10-07T00:00Z
source: hres_sample
edit:
  type: relocate
  params:
    east_km: -100
    north_km: 0
taper:
  centre: [22.5, -93.4]
  radius_km: 400.0
seeds: {baseline: 0, control: 1000}
```

Scale the intensity, with a factor between 0.5 and 1.5:

```yaml
id: my_scaling
title: Hurricane Milton, intensity scaled to 1.2 of the extracted anomaly
kind: fork
storm: 2024279N21265
init: 2024-10-07T00:00Z
source: hres_sample
edit:
  type: scale_intensity
  params:
    factor: 1.2
taper:
  centre: [22.5, -93.4]
  radius_km: 400.0
seeds: {baseline: 0, control: 1000}
```

Change the sea surface temperature, by at most 3 K either way:

```yaml
id: my_warm_gulf
title: Hurricane Milton, Gulf of Mexico sea surface warmed by 1 K at its peak
kind: fork
storm: 2024279N21265
init: 2024-10-07T00:00Z
source: hres_sample
edit:
  type: sst_anomaly
  params:
    centre: [25.5, -86.0]
    sigma_km: 400
    delta_k: 1
seeds: {baseline: 0, control: 1000}
```

Add noise, with an amplitude between 0 and 0.05:

```yaml
id: my_noise
title: Hurricane Milton, noise at 2 percent of the spatial standard deviation
kind: fork
storm: 2024279N21265
init: 2024-10-07T00:00Z
source: hres_sample
edit:
  type: noise
  params:
    amplitude: 0.02
    seed: 3
seeds: {baseline: 0, control: 1000}
```

Take one parameter at several values, one fork per value:

```yaml
id: my_scaling_sweep
title: Hurricane Milton, intensity scaled from 0.5 to 1.5
kind: sweep
storm: 2024279N21265
init: 2024-10-07T00:00Z
source: hres_sample
edit:
  type: scale_intensity
  params: {}
sweep:
  param: factor
  values: [0.5, 0.75, 1.0, 1.25, 1.5]
taper:
  centre: [22.5, -93.4]
  radius_km: 400.0
seeds: {baseline: 0, control: 1000}
```

## Run your own

Save a recipe as `my.yaml` and run

```sh
uv run stormfork rerun my.yaml --members 1 --steps 8
```

The command checks the recipe, runs the baseline, fork and control, and writes the manifest, atlases and track files to `./<id>/`, or to `--out`. It needs the model dependency group, which `uv sync` installs, and a recipe whose `source` is `hres_sample`; for an ERA5 start, use `notebooks/rerun_experiment.ipynb`. A recipe has no published run to compare with, so nothing is verified.

On first use `rerun` downloads the HRES sample, the model weights and IBTrACS into a local cache; `stormfork fetch` does this ahead of time, and `--sample`, `--weights` and `--ibtracs` take local files instead. [Reproducibility](reproducibility.md#reproducing-a-published-experiment) lists what each step writes.

## Reproduce a published experiment

For an experiment that starts from the HRES sample:

```sh
uv run stormfork rerun milton_south_75km --by <your-github-login>
```

This reruns it and compares its tracks with the published ones through `stormfork verify --reproduction`. The result is one JSON entry that passes when the largest track difference is within the limit for your runtime. The other experiments start from ERA5; reproduce those in the notebook. [Reproducibility](reproducibility.md) has the tolerance, the ledger and how a reproduction is recorded.

## Propose an experiment

1. Write the recipe and check it loads: save it as `<id>.yaml` in an empty folder and run `uv run stormfork scenarios list --dir <folder>`.
2. Run it and export the run directory, with `rerun` or the notebook.
3. Upload the zip as a pull request on the experiment dataset, with your GitHub login as `run_by` in the manifest. [Reproducibility](reproducibility.md#contributor-proposed-experiments) has the command.
4. Ask someone else to reproduce it and add their entry to the reproduction ledger. A maintainer publishes the experiment only after a passing reproduction by someone other than `run_by`.

State the limits you know of in the recipe's `limits`. A storm before 2024 must say it is inside the model's training period, and a hindcast must say its initial conditions are reconstructed from ERA5.

## Use the edits without the model

The edits are plain functions on xarray datasets. This example opens any regular-grid dataset, builds a `Taper`, calls `relocate` and then `validate`. It runs on a 0.25 degree grid; the `names` mapping tells the edit which variable holds mean sea-level pressure.

```python
import xarray as xr

from stormfork.perturb import Taper, relocate, validate

# Any regular grid works: ascending lat and lon with equal spacing, lon from 0 to 360.
ds = xr.open_dataset("grid.nc").load()
names = {"mean_sea_level_pressure": "mslp"}
taper = Taper(centre_lat=25.0, centre_lon=280.0, radius_km=900.0)
moved = relocate(ds, taper, east_km=180.0, north_km=240.0, names=names)
# Raises PhysicalBoundsError if the edited state leaves the chosen bounds.
validate(ds, moved, names=names)
for label, data in (("before", ds), ("after", moved)):
    low = data["mslp"].argmin(dim=("lat", "lon"))
    print(label, float(data.lat[low["lat"]]), float(data.lon[low["lon"]]))
moved.to_netcdf("moved.nc")
```

`validate` raises `PhysicalBoundsError` for a state outside the chosen bounds. Pass `moved_taper=` with the destination taper to also check the land fraction of the relocated core, which needs a `land_sea_mask` variable. The [API notes](adr/0012-perturb-public-api.md) give the grid contract and the units.
