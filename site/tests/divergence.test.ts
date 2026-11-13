import { describe, expect, it } from "vitest";

import { divergenceText, parseDivergence, sparklinePaths } from "../src/divergence";

const REPORT = {
  grid_resolution_deg: 1,
  steps: [
    { step: 0, valid_time: "2026-01-01T00:00:00Z", fork_sep_km: 10, control_sep_km: 5 },
    { step: 1, valid_time: "2026-01-01T06:00:00Z", fork_sep_km: null, control_sep_km: 2.4 },
  ],
};

describe("parseDivergence", () => {
  it("keeps the two separations and their step", () => {
    expect(parseDivergence(REPORT)).toEqual([
      { step: 0, forkKm: 10, controlKm: 5 },
      { step: 1, forkKm: null, controlKm: 2.4 },
    ]);
  });

  it("rejects a report without steps or with a bad separation", () => {
    expect(() => parseDivergence({})).toThrow("not a divergence report");
    expect(() => parseDivergence({ steps: [{ step: 0, fork_sep_km: -1 }] })).toThrow(
      "invalid separation",
    );
    expect(() => parseDivergence({ steps: [{ step: -1 }] })).toThrow("malformed step");
  });
});

describe("divergenceText", () => {
  it("always reads the edit beside the control", () => {
    expect(divergenceText({ step: 0, forkKm: 42, controlKm: 18 })).toBe(
      "Fork separation 42 km; control separation 18 km.",
    );
    expect(divergenceText({ step: 1, forkKm: null, controlKm: 18 })).toBe(
      "No fork separation is published; control separation 18 km.",
    );
    expect(divergenceText({ step: 1, forkKm: 42, controlKm: null })).toBe(
      "Fork separation 42 km; no control separation is published.",
    );
    expect(divergenceText({ step: 1, forkKm: null, controlKm: null })).toBe(
      "No separation is published for this step.",
    );
    expect(divergenceText(null)).toBe("No divergence is published for this experiment.");
  });
});

describe("sparklinePaths", () => {
  it("puts both series on one shared scale", () => {
    const paths = sparklinePaths(parseDivergence(REPORT), 100, 10);
    expect(paths.maxKm).toBe(10);
    expect(paths.fork).toBe("0.00,0.00");
    expect(paths.control).toBe("0.00,5.00 100.00,7.60");
  });

  it("keeps a flat series from dividing by zero", () => {
    expect(sparklinePaths([{ step: 0, forkKm: 0, controlKm: 0 }], 100, 10)).toEqual({
      fork: "0.00,10.00",
      control: "0.00,10.00",
      maxKm: 1,
    });
  });
});
