import { describe, expect, it } from "vitest";

import { parseReproductions, reproductionText } from "../src/reproductions";

const ENTRY = {
  by: "alice",
  runtime: "cpu: Darwin arm64",
  date: "2026-11-06",
  code_version: "v1",
  max_track_diff_km: 0.5,
  passed: true,
};

describe("parseReproductions", () => {
  it("keeps who, where and whether it passed", () => {
    expect(parseReproductions([ENTRY])).toEqual([
      { by: "alice", runtime: "cpu: Darwin arm64", passed: true },
    ]);
  });

  it("rejects a document that is not a list of entries", () => {
    expect(() => parseReproductions({})).toThrow("not a list");
    expect(() => parseReproductions([{ by: "alice" }])).toThrow("malformed entry");
    expect(() => parseReproductions([null])).toThrow("malformed entry");
  });
});

describe("reproductionText", () => {
  it("says nothing until a rerun has passed", () => {
    expect(reproductionText([])).toBeNull();
    expect(reproductionText([{ by: "alice", runtime: "cpu", passed: false }])).toBeNull();
  });

  it("counts one person on one runtime", () => {
    expect(reproductionText([{ by: "alice", runtime: "cpu", passed: true }])).toBe(
      "Reproduced by 1 person on 1 runtime",
    );
  });

  it("counts distinct people and runtimes, ignoring failures and login case", () => {
    expect(
      reproductionText([
        { by: "alice", runtime: "cpu", passed: true },
        { by: "Alice", runtime: "gpu", passed: true },
        { by: "bob", runtime: "cpu", passed: true },
        { by: "carol", runtime: "tpu", passed: false },
      ]),
    ).toBe("Reproduced by 2 people on 2 runtimes");
  });
});
