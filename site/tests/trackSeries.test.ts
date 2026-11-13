import { describe, expect, it } from "vitest";
import { seriesFromTracks } from "../src/trackSeries";
import type { TrackSample } from "../src/tracks";

function samples(steps: readonly number[]): TrackSample[] {
  return steps.flatMap((step) =>
    Array.from({ length: 5 }, (_, member) => ({
      run: "fork",
      member,
      step,
      lat: 20 + step,
      lon: -60 + member,
    })),
  );
}

describe("track series", () => {
  it("splits every member and mean at a missing step", () => {
    const series = seriesFromTracks(samples([3, 0, 2, 1, 6, 5]));
    const members = series.filter((line) => line.weight === 1);
    const means = series.filter((line) => line.weight === 3);
    expect(members).toHaveLength(10);
    expect(means).toHaveLength(2);
    for (const line of series) {
      expect(line.points.map((point) => point.lat)).toEqual(
        line.points[0]?.lat === 20 ? [20, 21, 22, 23] : [25, 26],
      );
    }
    expect(means.flatMap((line) => line.markers ?? []).map((point) => point.lat)).toEqual([
      20, 22, 26,
    ]);
  });

  it("leaves isolated fixes disconnected and preserves member/run separation", () => {
    const input = [...samples([0, 2]), { run: "baseline", member: 0, step: 1, lat: 50, lon: 0 }];
    const series = seriesFromTracks(input);
    expect(series).toHaveLength(13);
    expect(series.every((line) => line.points.length === 1)).toBe(true);
    expect(input.map((point) => point.step)).toEqual([0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1]);
  });

  it("splits a mean when too few members remain for an intermediate step", () => {
    const input = samples([0, 1, 2]).filter((point) => point.step !== 1 || point.member < 4);
    const means = seriesFromTracks(input).filter((line) => line.weight === 3);
    expect(means).toHaveLength(2);
    expect(means.map((line) => line.points.length)).toEqual([1, 1]);
  });
});
