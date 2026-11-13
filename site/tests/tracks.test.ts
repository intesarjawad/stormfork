import { describe, expect, it } from "vitest";

import {
  forecastSamples,
  lonLatToPosition,
  observedTrack,
  parseTracks,
  trackPositions,
  wrapLongitude,
} from "../src/tracks";

function expectPosition(
  actual: readonly [number, number, number],
  expected: readonly [number, number, number],
): void {
  expect(actual[0]).toBeCloseTo(expected[0], 12);
  expect(actual[1]).toBeCloseTo(expected[1], 12);
  expect(actual[2]).toBeCloseTo(expected[2], 12);
}

describe("track geometry", () => {
  it("places the cardinal points where the atlas seam expects them", () => {
    expectPosition(lonLatToPosition(0, 0), [-1, 0, 0]);
    expectPosition(lonLatToPosition(0, 90), [0, 0, 1]);
    expectPosition(lonLatToPosition(0, 180), [1, 0, 0]);
    expectPosition(lonLatToPosition(90, 0), [0, 1, 0]);
    expectPosition(lonLatToPosition(-90, 0), [0, -1, 0]);
  });

  it("scales by radius and wraps longitudes into one turn", () => {
    expectPosition(lonLatToPosition(0, 370, 2), lonLatToPosition(0, 10, 2));
    expectPosition(lonLatToPosition(0, -10, 2), lonLatToPosition(0, 350, 2));
    expect(lonLatToPosition(0, 10, 2)[1]).toBeCloseTo(0, 12);
  });

  it("flattens a track into xyz positions", () => {
    const flat = trackPositions(
      [
        { lat: 0, lon: 0 },
        { lat: 0, lon: 90 },
      ],
      1,
    );
    expect(flat).toHaveLength(6);
    [-1, 0, 0, 0, 0, 1].forEach((expected, index) => {
      expect(flat[index]).toBeCloseTo(expected, 12);
    });
  });

  it("wraps longitudes", () => {
    expect(wrapLongitude(0)).toBe(0);
    expect(wrapLongitude(360)).toBe(0);
    expect(wrapLongitude(-10)).toBe(350);
    expect(wrapLongitude(725)).toBe(5);
  });
});

describe("track records", () => {
  const SAMPLE = { run: "baseline", member: 0, step: 4, lat: 20, lon: -60 };

  it("accepts published samples", () => {
    expect(parseTracks([SAMPLE])).toEqual([SAMPLE]);
  });

  it("rejects malformed records", () => {
    expect(() => parseTracks({})).toThrow("not an array");
    expect(() => parseTracks([{ ...SAMPLE, lat: 91 }])).toThrow("malformed");
    expect(() => parseTracks([{ ...SAMPLE, run: "" }])).toThrow("malformed");
    expect(() => parseTracks([{ ...SAMPLE, member: 1.5 }])).toThrow("malformed");
  });
});

describe("observed track", () => {
  const sample = (run: string, member: number, step: number) => ({
    run,
    member,
    step,
    lat: 20 + step,
    lon: -60,
  });
  const MIXED = [
    sample("baseline", 0, 0),
    sample("observed", 0, 3),
    sample("baseline", 1, 0),
    sample("observed", 0, 0),
    sample("observed", 0, 1),
  ];

  it("returns the observed samples in step order", () => {
    expect(observedTrack(MIXED).map((item) => item.step)).toEqual([0, 1, 3]);
    expect(observedTrack(MIXED).every((item) => item.run === "observed")).toBe(true);
  });

  it("keeps the forecast runs apart from the truth", () => {
    expect(forecastSamples(MIXED)).toEqual([sample("baseline", 0, 0), sample("baseline", 1, 0)]);
  });

  it("is empty when no observed run is published", () => {
    const forecast = [sample("baseline", 0, 0)];
    expect(observedTrack(forecast)).toEqual([]);
    expect(forecastSamples(forecast)).toEqual(forecast);
  });

  it("is published as ordinary samples that parse like the forecast runs", () => {
    expect(parseTracks(MIXED)).toEqual(MIXED);
  });
});
