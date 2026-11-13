import type { TrackSeries } from "./globeRenderer";
import { ensembleMeans } from "./steps";
import type { TrackPoint, TrackSample } from "./tracks";

function continuous<T extends TrackPoint & { readonly step: number }>(points: readonly T[]): T[][] {
  const segments: T[][] = [];
  let previous = -2;
  for (const point of [...points].sort((a, b) => a.step - b.step)) {
    const segment = point.step === previous + 1 ? segments.at(-1) : undefined;
    if (segment === undefined) segments.push([point]);
    else segment.push(point);
    previous = point.step;
  }
  return segments;
}

export function seriesFromTracks(tracks: readonly TrackSample[]): readonly TrackSeries[] {
  const grouped = new Map<string, { run: string; points: TrackSample[] }>();
  for (const sample of tracks) {
    const key = `${sample.run}:${String(sample.member)}`;
    const entry = grouped.get(key) ?? { run: sample.run, points: [] };
    entry.points.push(sample);
    grouped.set(key, entry);
  }
  const members = [...grouped.values()].flatMap(({ run, points }) =>
    continuous(points).map((points) => ({ run, points, weight: 1, opacity: 0.45 })),
  );
  const means = [...new Set(tracks.map((sample) => sample.run))].flatMap((run) =>
    continuous(ensembleMeans(tracks, run)).map((points) => ({
      run,
      points,
      weight: 3,
      opacity: 1,
      ...(run === "fork" ? { markers: points.filter((point) => point.step % 2 === 0) } : {}),
    })),
  );
  return [...members, ...means];
}
