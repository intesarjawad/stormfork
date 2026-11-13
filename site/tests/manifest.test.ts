import { describe, expect, it } from "vitest";

import { isManifest } from "../src/manifest";

interface ExampleField {
  name: string;
  files: string[];
  range: [number, number];
  encoding: string;
  resolution_deg: number;
}

type ExampleRuns = Record<string, Record<string, string>>;

function exampleFields(): ExampleField[] {
  return [
    {
      name: "mslp",
      files: ["mslp"],
      range: [92000, 105000],
      encoding: "linear",
      resolution_deg: 1,
    },
    {
      name: "wind10",
      files: ["u10", "v10"],
      range: [-60, 60],
      encoding: "linear",
      resolution_deg: 1,
    },
    { name: "t2m", files: ["t2m"], range: [220, 320], encoding: "linear", resolution_deg: 1 },
    { name: "z500", files: ["z500"], range: [48000, 59000], encoding: "linear", resolution_deg: 1 },
    { name: "tp6h", files: ["tp6h"], range: [0, 0.1], encoding: "sqrt", resolution_deg: 1 },
  ];
}

function exampleRuns(): ExampleRuns {
  return {
    baseline: {
      "baseline/mslp.png": "a".repeat(64),
      "preview/baseline_mslp_f0.png": "b".repeat(64),
    },
    fork: { "fork/mslp.png": "c".repeat(64), "preview/fork_mslp_f0.png": "d".repeat(64) },
    control: {
      "control/mslp.png": "e".repeat(64),
      "preview/control_mslp_f0.png": "f".repeat(64),
    },
  };
}

function exampleManifest(
  fields: readonly ExampleField[] = exampleFields(),
  runs: ExampleRuns = exampleRuns(),
): Record<string, unknown> {
  return {
    id: "hindcast_synthetic",
    title: "Synthetic comparison",
    kind: "hindcast",
    steps: 40,
    members: 8,
    fields,
    runs,
    created: "2026-11-05T12:00:00Z",
  };
}

function withField(index: number, change: Partial<ExampleField>): ExampleField[] {
  return exampleFields().map((field, position) =>
    position === index ? { ...field, ...change } : field,
  );
}

describe("isManifest", () => {
  it("accepts a manifest written by the exporter", () => {
    expect(isManifest(exampleManifest())).toBe(true);
    expect(isManifest(exampleManifest(withField(1, { resolution_deg: 2 })))).toBe(true);
  });

  it.each(["id", "title", "fields", "runs"])("rejects a manifest without %s", (key) => {
    const manifest = Object.fromEntries(
      Object.entries(exampleManifest()).filter(([name]) => name !== key),
    );
    expect(isManifest(manifest)).toBe(false);
  });

  it.each([null, [], "manifest", 7])("rejects a non-object manifest", (value) => {
    expect(isManifest(value)).toBe(false);
  });

  it("rejects a wind entry that does not list exactly u10 and v10", () => {
    expect(isManifest(exampleManifest(withField(1, { files: ["u10"] })))).toBe(false);
    expect(isManifest(exampleManifest(withField(1, { files: ["v10", "u10"] })))).toBe(false);
    expect(isManifest(exampleManifest(withField(1, { files: ["u10", "v10", "w10"] })))).toBe(false);
  });

  it("rejects any resolution other than 1 or 2 degrees", () => {
    expect(isManifest(exampleManifest(withField(1, { resolution_deg: 3 })))).toBe(false);
    expect(isManifest(exampleManifest(withField(0, { resolution_deg: 0 })))).toBe(false);
  });

  it("rejects a field the site cannot decode", () => {
    expect(isManifest(exampleManifest(withField(0, { files: [] })))).toBe(false);
    expect(isManifest(exampleManifest(withField(0, { files: ["mslp", "t2m"] })))).toBe(false);
    expect(isManifest(exampleManifest(withField(0, { encoding: "log" })))).toBe(false);
    expect(isManifest(exampleManifest(withField(0, { range: [105000, 92000] })))).toBe(false);
  });

  it("rejects a manifest without runs or with malformed assets", () => {
    expect(isManifest(exampleManifest(exampleFields(), {}))).toBe(false);
    expect(isManifest(exampleManifest(exampleFields(), { baseline: {} }))).toBe(false);
    expect(
      isManifest(exampleManifest(exampleFields(), { baseline: { "baseline/mslp.png": "" } })),
    ).toBe(false);
    expect(isManifest(exampleManifest(exampleFields(), { "": { "baseline/mslp.png": "a" } }))).toBe(
      false,
    );
  });

  it("reads only its own schema version", () => {
    expect(isManifest({ ...exampleManifest(), schema_version: 2 })).toBe(true);
    expect(isManifest({ ...exampleManifest(), schema_version: 1 })).toBe(false);
    expect(isManifest({ ...exampleManifest(), schema_version: "2" })).toBe(false);
  });

  it("accepts a storm without a position at the init and nothing else null", () => {
    const withStorm = (storm: Record<string, unknown>): Record<string, unknown> => ({
      ...exampleManifest(),
      init: {
        source: "era5",
        time: "2024-10-04T00:00:00",
        storm: {
          ibtracs_sid: "2024279N21265",
          lat: 22.5,
          lon: 266.6,
          first_record_time: "2024-10-05T12:00:00",
          ...storm,
        },
      },
    });
    expect(isManifest(withStorm({}))).toBe(true);
    expect(isManifest(withStorm({ lat: null, lon: null }))).toBe(true);
    expect(isManifest(withStorm({ lat: null }))).toBe(false);
    expect(isManifest(withStorm({ lon: null }))).toBe(false);
    expect(isManifest(withStorm({ ibtracs_sid: null }))).toBe(false);
    expect(isManifest(withStorm({ first_record_time: null }))).toBe(false);
    expect(isManifest(withStorm({ lat: 91 }))).toBe(false);
    expect(isManifest({ ...exampleManifest(), init: null })).toBe(false);
  });

  it("accepts a sweep landfall step that is a step index and rejects anything else", () => {
    const sweep = (landfall: unknown): Record<string, unknown> => ({
      ...exampleManifest(),
      sweep: {
        name: "displacement",
        baseline: "none",
        landfall_step: landfall,
        stops: [
          { id: "none", direction: "none", distance_km: 0, tracks: "stops/none/tracks.json" },
        ],
      },
    });
    expect(isManifest(sweep(20))).toBe(true);
    expect(isManifest(sweep(undefined))).toBe(true);
    expect(isManifest(sweep(-1))).toBe(false);
    expect(isManifest(sweep(1.5))).toBe(false);
    expect(isManifest(sweep("20"))).toBe(false);
  });

  it("accepts the sweep block the exporter writes, or null without one", () => {
    const digest = "a".repeat(64);
    const exported = {
      name: "east_km",
      baseline: "none_0",
      stops: [
        {
          id: "none_0",
          direction: "none",
          distance_km: 0,
          tracks: "tracks.json",
          atlas: "fork/mslp.png",
          divergence: "divergence.json",
          sha256: {},
        },
        {
          id: "east_150",
          direction: "east",
          distance_km: 150,
          tracks: "stops/east_km_150/tracks.json",
          atlas: "stops/east_km_150/mslp.png",
          divergence: "stops/east_km_150/divergence.json",
          sha256: {
            "stops/east_km_150/tracks.json": digest,
            "stops/east_km_150/mslp.png": digest,
            "stops/east_km_150/divergence.json": digest,
          },
        },
      ],
    };
    const withSweep = (sweep: unknown): Record<string, unknown> => ({
      ...exampleManifest(),
      sweep,
    });
    expect(isManifest(withSweep(exported))).toBe(true);
    expect(isManifest(withSweep(null))).toBe(true);
    expect(isManifest(withSweep({ ...exported, baseline: "west_150" }))).toBe(false);
    const [none, east] = exported.stops;
    expect(isManifest(withSweep({ ...exported, stops: [none, { ...east, sha256: [] }] }))).toBe(
      false,
    );
    expect(
      isManifest(withSweep({ ...exported, stops: [none, { ...east, sha256: { a: 1 } }] })),
    ).toBe(false);
  });
});
