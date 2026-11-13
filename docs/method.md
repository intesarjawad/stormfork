# Method

I fork a forecast: change one thing in a storm's starting state, run the model again and measure what moved. This page says what is compared and what that comparison can and cannot support. Results for each published experiment come from its exported tracks and divergence files.

## An experiment is three runs

Every fork or sweep experiment starts from one initialization time and one initial state, and runs the model three times:

| Run | Starting state | Random seeds |
|---|---|---|
| baseline | unedited | seed S |
| fork | edited | the baseline's seeds |
| control | unedited | the baseline's seeds plus 1000 |

The fork shares the baseline's seeds, so the two differ only by the edit. The control differs from the baseline only by its seeds. It shows how far the model's own variability moves a forecast, so an edit is read against it and not against zero. Every experiment file fixes the baseline seed at 0 and the control seed at 1000. Each run is an ensemble of members. A sweep repeats the fork for several values of one edit parameter, and a hindcast has a baseline and the observed track but no edit. The files are in `experiments/` and `stormfork scenarios list` prints them.

Small differences in a starting state can grow into larger ones as a forecast runs. [Selz and Craig](https://doi.org/10.1002/2015GL063525) study that upscale growth for convective uncertainty in a regional model, not in the model I use. I cite it for the idea alone: a forecast that differs from another is not yet evidence of an effect, which is why the control exists.

## Starting states

The model takes analyses on a one-degree grid with 13 pressure levels. Each experiment names its source:

- `hres_sample` is the HRES sample that ships with the model, for 2024-10-07 00 UTC.
- `era5` builds the same fields from the [ERA5 reanalysis](https://doi.org/10.1002/qj.3803) for any other date.

ERA5 is a different analysis system from HRES. [ADR 0002](adr/0002-initial-condition-source.md) records why ERA5 is the source for hindcasts and where that choice is weak. A hindcast therefore starts from a reconstructed state, and its limits say so: "Initial conditions are reconstructed from ERA5 reanalysis, not the analysis available at the time, so a hindcast shows what the model does from a reconstructed state and is not the forecast it would have issued then."

## The edit

An edit changes the starting state inside a circular taper around the storm. [Perturbations](perturbations.md) gives the units, the chosen limits and the figures below. The four edit types are:

| Type | Parameters | Effect |
|---|---|---|
| `relocate` | `east_km`, `north_km` | moves the storm anomaly by up to 600 km |
| `scale_intensity` | `factor` | multiplies the storm anomaly by a factor from 0.5 to 1.5 |
| `sst_anomaly` | `centre`, `sigma_km`, `delta_k` | adds a Gaussian change of up to 3 K to sea surface temperature |
| `noise` | `amplitude`, `seed` | adds seeded noise of up to 0.05 of each slice's spatial standard deviation |

A taper's centre and radius come from the storm's best-track row at the initialization time, taken from [IBTrACS](https://doi.org/10.1175/2009BAMS2755.1): 1.5 times the largest reported gale-force radius, held between 400 and 900 km. `stormfork scenarios taper` prints the block for a storm and time. Before a run, `validate` rejects states outside chosen physical bounds. The bounds are screens for implausible states, not tested limits of the model.

## Forecast and tracks

The model is a WeatherNext Cyclones Mini checkpoint. It advances the state in six-hour steps. The tracker from the same package follows the storm in each member's output. For a relocation, the tracker starts from the moved position, so it follows the moved storm and not the original one. The tracker also follows every other storm IBTrACS lists at the init, anywhere on the globe; each member keeps only the track that starts within 200 km of the target storm's IBTrACS position (the moved position for a relocation), and a member with no such track is reported as having lost the storm. Every run goes through the same tracker, so a difference between runs is a difference in what it reports. A step where it loses the storm is a gap, never an interpolated position.

## What I compare

`divergence.json` holds these measures for every step. Distances are great-circle kilometres.

- Fork separation and control separation: the distance from the baseline ensemble-mean position to the fork's, and to the control's. A mean exists only at steps where at least five members have a matched track.
- Signal to noise: fork separation divided by the larger of the control separation and one grid cell. The floor stops a control that matches the baseline almost exactly from inflating the ratio.
- Baseline spread: the mean distance of the baseline's members from their mean.
- Paired test: for each member present in all three runs, the distance of its fork position from its baseline position minus the distance of its control position from its baseline position. A one-sided exact Wilcoxon signed-rank test ([Wilcoxon](https://doi.org/10.2307/3001968)) asks whether the fork is farther away than the control. It needs five pairs. Zero differences are discarded before ranking; tied absolute differences receive midranks. The p-value counts all sign permutations of those actual ranks, including the observed sum. A sweep applies the Holm correction across its edited values at each step, and leaves the zero edit out of that family.
- Pressure change: the root-mean-square difference between ensemble-mean sea-level pressure fields within 1500 km of the baseline position.
- Intensity: the mean central pressure and peak wind of the matched members, and the 24-hour drift, the fork-minus-baseline pressure difference at 24 hours minus the same difference at the start.
- Adjustment ratio: the fork's root-mean-square pressure change over the first six hours divided by the baseline's, in the taper annulus. See [limits](perturbations.md#limits).

A sweep also carries a null-fork check. Its zero edit shares the baseline's seeds and changes nothing, so its fork must land on the baseline. The manifest records the largest member or ensemble-mean distance between the two as `diagnostics.null_fork_max_track_diff_km`, and `stormfork verify` fails a sweep that omits it, whose recorded value does not follow from the tracks, or that exceeds 10 km, the same-runtime limit in [Reproducibility](reproducibility.md).

The site's displacement chart plots each fork member's latitude at the landfall step against how far the storm was moved, one column per published stop, with the lowest to highest control latitude as a band. The manifest's `sweep.landfall_step` names that step; without it each member's last sample is used.

Observed tracks come from IBTrACS and appear only for hindcasts, scored against the National Hurricane Center's forecast aids as described in the README.

## Training period

The model's training data run through 2023. A storm before 2024 is inside that period, so its baseline is not an independent forecast. Every such experiment, including those for Sandy and Katrina, says so: "This storm is inside the model's training period (data through 2023), so the baseline is not an independent forecast; the fork shows how the trained model responds to the edit." Storms from 2024 on are outside it. I report both kinds, and report them whichever way they turn out.

## What this does not show

- It does not show forecast skill. A fork compares the model with itself under an edit.
- It does not show what the real storm would have done. The model's response to an edit is not the atmosphere's.
- A difference larger than the control is evidence that the edit mattered to the model, not that the model is right about it.
- A small ensemble is a small sample. Each step reports how many members were paired, and the test needs at least five.
- The edit limits and the adjustment ratio are chosen working limits. I have not measured how sensitive the results are to them.

## References

- Hersbach, H. et al. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society 146, 1999-2049. <https://doi.org/10.1002/qj.3803>
- Knapp, K. R., Kruk, M. C., Levinson, D. H., Diamond, H. J. and Neumann, C. J. (2010). The International Best Track Archive for Climate Stewardship (IBTrACS). Bulletin of the American Meteorological Society 91, 363-376. <https://doi.org/10.1175/2009BAMS2755.1>
- Selz, T. and Craig, G. C. (2015). Simulation of upscale error growth with a stochastic convection scheme. Geophysical Research Letters 42, 3056-3062. <https://doi.org/10.1002/2015GL063525>
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics Bulletin 1, 80-83. <https://doi.org/10.2307/3001968>
