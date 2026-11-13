# 0014: What a run records: the adjustment ratio, early inits, hindcast ids and reruns

Status: accepted.

## Context

Four things a forecast run writes were left open, and each one decides a field
of the manifest or a folder on the site.

1. The adjustment ratio compares the fork's first-step pressure change with
   the baseline's in the taper annulus. Noise and SST edits have no taper, so
   they had no ratio, and a relocation was measured only where the storm
   started.
2. A hindcast init can come before the storm exists. Milton's first init,
   2024-10-04 00 UTC, precedes its first IBTrACS record, and `run` refused it
   because the manifest needed the storm's position at the init.
3. A hindcast has three inits, but the manifest has one id, one init and one
   folder.
4. `rerun` verifies a reproduction against the published run, and that result
   could have gone into the rerun's manifest.

## Decision

**Every fork reports the same ratio.** A tapered edit keeps the definition:
the annulus from half the radius to the radius, around the edit. An edit
without a taper is measured over the same annulus geometry, centred on the
storm's IBTrACS position at the init, with the radius the taper rule gives
that storm there: 1.5 times its largest gale-force radius, held between 400
and 900 km. A tapered and an untapered fork of one storm are then measured
over the same ring. A storm with no gale-force radius recorded at the init
gets the rule's smallest radius, 400 km. A relocation disturbs the field where the storm was and
where it lands, so it is measured around both and the larger value is
recorded. The manifest keeps one number per fork, and a sweep records each
stop's in `stops.json`. Before the storm's first record there is no position
to centre on, so an untapered fork there has no ratio.

I chose one quantity over a comparable region so the publication gate means
the same thing for every edit. A noise or SST edit still disturbs the storm's
surroundings, and leaving it unmeasured would let those forks pass the gate
by default. Taking the larger of the two relocation values keeps the gate
from missing a shock at the destination.

**An init before genesis is valid.** The manifest's storm position at the init
is null there, with the storm's first-record time beside it. Nothing else in
the manifest may be null in its place: the validator accepts a null position
only together with a first record after the init, and requires a position at
or after it. The schema version is now 2, a field every manifest carries. The
Python writer and validator and the site's manifest guard accept the null
position and reject every other null in the storm record, and the site reads
only version 2.

The consumers of the position handle its absence explicitly:

- `run` (`storm_at_init`) records the null position and the first record
  instead of refusing the init.
- The adjustment ratio of an untapered fork reads the position, and records
  no ratio without it.
- The tracker never read the manifest's position. It seeds from IBTrACS, and a
  storm that forms after the init is matched to its observed track.
- The site's manifest guard validates the record but draws nothing from it,
  and no plot script reads it.

No release has been published, so there is no version 1 manifest to migrate.

**Each hindcast init is its own experiment.** Its id is the hindcast's id with
the init time as a `-YYYYMMDDHH` suffix, such as
`hindcast_milton_2024-2024100400`, and the manifest records the hindcast's id
as `parent_id`. Each init then has its own folder, release zip and lock entry,
and `verify` checks it like any other experiment. Recipe ids cannot contain a
hyphen, so the suffix is never ambiguous. The validator requires the id to
match the parent and the init, and allows a parent only on a hindcast.

**A rerun's verify outcome stays out of the manifest.** The manifest describes
how a run was made, and its hashes pin the published files. Whether someone
else reproduced it is a later fact about the run, recorded in the reproduction
ledger, `reproductions.json`, by the person who reproduced it. Writing it into
the manifest would change the bytes a release already pinned.

## Evidence and limits

Tests on synthetic fields check the ratio around the original and the moved
annulus, around the storm at the rule's radius for an untapered edit, the
400 km fallback without a gale-force radius and its absence before genesis; a stand-in model runs a pre-genesis hindcast init, a sweep with a
refused stop and a fork that records its ratio, land fraction and bounds
check. Measuring untapered edits over the taper rule's ring is a choice for
comparability, not a calibrated scale of adjustment. Real Milton forecasts have
measured ratios of 1.8029 for the 150 km fork and 1.4742 for the shipped 75 km
fork; [Reproducibility](../reproducibility.md#a-full-experiment-on-eden) records
their runtimes and member counts. Those measurements do not calibrate the
publication threshold across storms or edit types.
