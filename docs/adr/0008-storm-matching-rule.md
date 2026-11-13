# 0008: Which track is the target storm

Status: accepted.

## Context

The tracker follows every cyclone in a member's forecast and numbers each one.
An experiment needs one track per member: the storm that was edited or hindcast.
The comparison between runs fails if a member loses the storm for a reason
unrelated to the edit.

## Decision

**A storm that exists at the init** (every fork and control, and a hindcast whose
init follows genesis) is the track whose step-0 position lies within 200 km of
the storm's initial position, moved by the displacement for a relocation fork.
Its later path is never compared with the observed track, so a fork that sails
away from the observed storm stays matched, which is the result the site shows.

**A storm that forms after the init** is the predicted track with the smallest
mean great-circle distance to the IBTrACS track over their common valid times,
among tracks with at least four common times, all within 500 km.

A member with no qualifying track has no track for the storm.

**Continuity across a gap.** The tracker ends a track when it misses a
six-hourly fix and starts a new one under a new id if it finds the storm again.
In the Sandy baseline, one member lost the storm only for that reason. A
matched track therefore continues into another track of the same member when
that track starts one or two steps after it ends (at most one missing fix) and
its first fix lies within 200 km of the position extrapolated linearly from the
matched track's last two fixes. With several candidates the nearest wins, and
the rule repeats from the new end, so a track can bridge more than one gap. The
joined rows keep the first track's id, so a member still has one target track.
A gap of two or more missing fixes ends the track.

I took the gap limit and the distance test from TempestExtremes' `StitchNodes`,
which links detections across at most `maxgap` missing times within a maximum
distance (Ullrich et al., 2021, Geosci. Model Dev. 14, 5023). The distance is the
200 km the rule already uses to identify the storm at the init, so no new
threshold is introduced. Extrapolating from the last two fixes, rather than
comparing with the last fix, is my addition: over a 12 h gap a hurricane moves
several hundred kilometres, and a threshold on the raw distance would either
reject the real continuation or accept an unrelated storm.

**Continuity through extratropical transition.** The tracker follows the model's
cyclone-existence probability and ends a track when the mean probability within
500 km falls below 1e-5. A hurricane that turns extratropical keeps a deep low
but the probability collapses, and in the Sandy baseline the matched track ended
between steps 8 and 23 while the low persisted, so five or fewer of eight members
survived past step 12. After the rules above pick the target track, it continues
for as long as the member's mean sea level pressure holds a closed minimum near
it. At each next step the candidate is the pressure minimum nearest the position
extrapolated from the last two fixes, within 400 km, whose pressure is at least
200 Pa below the mean on a ring 5.5 degrees (600 km) away. Its position is the
vertex of a parabola through the grid minimum and its neighbours along each axis,
so continued fixes keep the sub-grid precision of the tracker's own instead of
snapping to whole degrees. The refined position must still be within the 400 km
search radius. The track ends at the first step with no candidate, at
the last forecast step, or when the candidate is a takeover. Previous and
current closed minima are first matched one-to-one by minimum total great-circle
distance, using SciPy's [linear sum assignment](https://github.com/scipy/scipy/blob/v1.18.1/doc/source/tutorial/optimize.rst).
A neighbouring source may move at most 400 km. The identified own source can
move farther only toward the selected candidate, which already passed the
400 km test around the own storm's extrapolation. Every other assignment is
still bounded to 400 km. Invalid pairs receive a cost greater than the sum of
all valid pair costs, then are discarded, so the assignment maximizes valid
matches before minimizing distance. The own source is the deepest previous
minimum within 200 km of the last fix.

If that own source is assigned to a different surviving minimum, the track
ends rather than jumping to the candidate. A different source already assigned
to another minimum cannot count as a merger into the candidate. An unmatched
source can count as merging only into its nearest current minimum, within
400 km. The candidate is a takeover if a linked neighbouring source was deeper
than the own source at the previous step. A low drawing in a shallower neighbour
keeps its track. This preserves distinct co-moving lows and prevents a stalled
own low from handing its track to a shallower neighbour.

Let r be the larger of 400 km and the candidate's distance from the last fix
plus the own-low tolerance. The current-field lookup extends 2r from the
candidate. The previous-field lookup extends 2r + 400 km, so every current
competitor can have its prior counterpart included within the drift bound.
Otherwise a persistent low outside the smaller previous window looks newborn
and can steal a neighbouring source's assignment.
If the own minimum is missing, at least one linked source must have been within
400 km of the last fix, and none may have come from beyond it. This permits a
displaced tracker fix while rejecting a bystander or an incoming system.
Missing pressure is never treated as an infinitely shallow low. A handover
with no previous field or no previous closed minimum ends the track. Rows added this way keep the track id, take pressure from the low's own grid minimum of sea level
pressure, and wind from the USA wind field at the new position, interpolating
cyclically across the 360-degree longitude seam. The USA pressure
field the tracker reads describes a tropical cyclone; over a decaying low it read
as much as 44 hPa deeper than the low itself.

This follows how TempestExtremes tracks extratropical systems: DetectNodes finds
sea level pressure minima with a closed-contour test (`msl,200.0,5.5,0`) and
requires no warm core, and StitchNodes links them (Zarzycki and Ullrich, 2017,
Geophys. Res. Lett. 44, 1141; Ullrich et al., 2021). New tracks, and the start
of every track, still come from the tracker's existence criterion, so a tropical
or non-tropical low is never promoted to a new storm. The 400 km radius is wider
than the 200 km used for gaps because the wobble in a decaying storm's last fix
left the true minimum 370 km from the extrapolated position in one baseline
member, which a 200 km test would have missed. The existence-probability threshold
and the verification statistics are unchanged.

### Own-low tolerance sensitivity

The 200 km own-low tolerance is an identity heuristic, not a calibrated physical
scale or a threshold supplied by TempestExtremes. On 2026-11-11 I retracked the
stored Sandy forecasts (8 members, 40 six-hour steps, init 2012-10-26 00 UTC),
using identical upstream detections and matches, with all four tracker fixes
above. Only the own-low tolerance changed. No forecast model was rerun.

| Own-low tolerance (km) | Baseline / fork / control rows | Step 16 fork separation (km) | Step 22 fork separation (km) | Step 22 pairs / p |
|---|---|---:|---:|---|
| 100 | 187 / 321 / 180 | 1260.577 | 3058.161 | 3 / null |
| 200 | 187 / 321 / 180 | 1260.577 | 3058.161 | 3 / null |
| 300 | 188 / 321 / 180 | 1260.577 | 3091.009 | 4 / null |

The 100 and 200 km tracks are identical. At 300 km, baseline member 4 extends
from step 21 to 22 and control member 7 from 24 to 25; control member 4
shortens from step 23 to 22. I retain 200 km, but
this single-storm sensitivity check establishes neither its optimality nor its
accuracy on other storms. At step 16 (30 Oct 00 UTC), all settings have eight
pairs and p=0.00390625. Against the rule before this fix, rows change from
187/322/188 to 187/321/180. Fork member 4 loses step 37; eight control fixes are removed.
Step 21 now has five pairs and p=0.03125 instead of seven and p=0.0078125.
Step 22 drops from five pairs and p=0.03125 to three pairs and no p-value.
The unchanged 1,261 km is at step 16, not step 22.

Reproduce without inference: run `stormfork track` on each stored member-field
dataset with the original init, target SID and relocation, then vary
`stormfork.track.transition.OWN_LOW_KM` over 100/200/300 on the same matched
tracks and pressure/wind fields. Export each set with
`stormfork.cli_export.export_run_directory`; statistics come from
`stormfork.export.divergence.compute_divergence`.

### Other retracks and pressure continuity

The same stored-output check on the older Milton 150 km fork changes matched
baseline/fork/control rows from 218/249/271 under the earlier rule to
217/232/250. At step 24 its fork separation changes from 264.275 to
429.477 km, pairs from five to four, and p=0.90625 becomes null. At step 25,
156.102 km with five pairs and p=0.84375 becomes an unpublished separation
with three pairs and no p-value. This is not the shipped 75 km recipe;
its forecast fields are absent, as are the other Milton edits, Katrina and
hindcast outputs. Their numbers remain unverified until those fields exist.

Pressure is intentionally a mixed diagnostic at handover: matched fixes use
the cyclone USA pressure field, while continued fixes use grid MSLP minima.
At the first continued fix, the same-position MSLP-minus-USA pressure difference
ranges from -16.01 to +3.62 hPa in Sandy (19 handovers), and -1.74 to +13.70 hPa
in the stored Milton 150 km run (15 handovers). Last-matched to first-continued
pressure jumps range from -14.62 to +10.23 hPa in Sandy and -9.66 to +11.20 hPa
in Milton. A jump can include the source change as well as six-hour evolution.
An ensemble intensity point can mix both definitions; it is not a homogeneous
central-pressure estimate. The separation and paired tests use positions.

## Consequences

- If forecast fields start at six hours and the matched track has only its
  init fix, it cannot be continued: there is no previous pressure field to
  establish identity.
- A member that loses the storm for longer than one fix still has a truncated
  track, and the ensemble mean needs five of eight members at each step.
- A track can outlive the storm as a decaying low, not a tropical cyclone. The
  association rule rejects an identifiable takeover by a previously deeper low.
  Before the takeover test, one Sandy control
  member filled over Ontario on 31 Oct and its track jumped to a separate low to
  the north, which it followed to Baffin Island and 980 hPa by 5 Nov; another went
  on to 58°N 52°W. Both now end before those later lows. Two members can end at
  different steps, and the ensemble mean still needs five of eight members.
- Two storms near each other that swap ids across a gap are indistinguishable by
  this rule. The nearest-to-extrapolation choice limits the error but does not
  remove it.
- Depth rank is not identity. The comparison uses the previous step's depth: a
  shallower neighbour that deepens later can inherit a track. Without an own
  minimum, a previous low already within the 400 km search region is ambiguous.
  Position-based assignment cannot resolve systems that exchange positions,
  and an unmatched source filling beside a survivor can resemble a merger.
  An unknown neighbour moving more than 400 km in six hours is not associated;
  its takeover may be missed. The own storm retains the extrapolation rule.
  These association and depth heuristics do not prove meteorological identity.
