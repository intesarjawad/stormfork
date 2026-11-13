import { existsSync, readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  ensembleMeans,
  formatKm,
  formatP,
  greatCircleKm,
  parseSteps,
  verdict,
  type StepStat,
} from "../src/steps";
import { fatesLine, storms, tally } from "../src/fates";
import { parseTracks } from "../src/tracks";

const DIRECTORY = new URL("../public/experiments/sandy_east_300km/", import.meta.url);

function read(name: string): unknown {
  return JSON.parse(readFileSync(new URL(name, DIRECTORY), "utf8"));
}

function stat(pairedP: number | null): StepStat {
  return { step: 0, validTime: "", forkKm: 1, controlKm: 1, pairedMembers: 8, pairedP };
}

describe("parseSteps", () => {
  it("keeps the separations and the paired test", () => {
    const [first] = parseSteps({
      steps: [
        {
          step: 3,
          valid_time: "2012-10-26T18:00:00",
          fork_sep_km: 424,
          control_sep_km: 41,
          paired_members: 8,
          paired_p: 0.0039,
        },
      ],
    });
    expect(first).toEqual({
      step: 3,
      validTime: "2012-10-26T18:00:00",
      forkKm: 424,
      controlKm: 41,
      pairedMembers: 8,
      pairedP: 0.0039,
    });
  });

  it("rejects a report that is not one and a probability above one", () => {
    expect(() => parseSteps({})).toThrow("not a divergence report");
    expect(() => parseSteps({ steps: [{ step: 0, paired_p: 1.5 }] })).toThrow("paired_p");
    expect(() => parseSteps({ steps: [{ step: 0, fork_sep_km: -2 }] })).toThrow("fork separation");
  });
});

describe("verdict", () => {
  it("reads the paired test against the 0.05 level and keeps a missing test apart", () => {
    expect(verdict(stat(0.0078))).toBe("separated");
    expect(verdict(stat(0.2))).toBe("within-noise");
    expect(verdict(stat(null))).toBe("too-few-pairs");
  });
});

describe("formatKm", () => {
  it("prints kilometres in one format and keeps a missing value missing", () => {
    expect(formatKm(3108.4)).toBe("3,108");
    expect(formatKm(null)).toBe("none");
  });
});

describe("formatP", () => {
  it("keeps the floor of an eight-member exact test readable", () => {
    expect(formatP(0.00390625)).toBe("0.0039");
    expect(formatP(0.0078125)).toBe("0.0078");
    expect(formatP(0.0156)).toBe("0.016");
    expect(formatP(0.0001)).toBe("< 0.001");
  });
});

const published = existsSync(new URL("tracks.json", DIRECTORY));

describe.skipIf(!published)("ensembleMeans on the published Sandy run", () => {
  // A skipped block is still collected, so read the gitignored run only when it exists.
  const tracks = published ? parseTracks(read("tracks.json")) : [];
  const steps = published ? parseSteps(read("divergence.json")) : [];

  it("reproduces the published fork separation from the member tracks", () => {
    const baseline = new Map(ensembleMeans(tracks, "baseline").map((point) => [point.step, point]));
    const fork = new Map(ensembleMeans(tracks, "fork").map((point) => [point.step, point]));
    let compared = 0;
    for (const stepStat of steps) {
      const a = baseline.get(stepStat.step);
      const b = fork.get(stepStat.step);
      if (a === undefined || b === undefined || stepStat.forkKm === null) {
        continue;
      }
      expect(greatCircleKm(a.lat, a.lon, b.lat, b.lon)).toBeCloseTo(stepStat.forkKm, -0.5);
      compared += 1;
    }
    expect(compared).toBeGreaterThan(20);
  });

  it("ends on a verdict whose fates match the ledger's heads", () => {
    const all = storms(tracks);
    const of = (run: string) => all.filter((storm) => storm.run === run);
    const last = Math.max(...steps.map((stat) => stat.step));
    const ashore = (run: string) => tally(of(run), last).ashore;
    expect(fatesLine(of("baseline"), of("fork"), last)).toBe(
      `${String(ashore("baseline"))} unedited ashore, ${String(ashore("fork"))} moved`,
    );
    expect(fatesLine(of("baseline"), of("fork"), last)).toBe("8 unedited ashore, 1 moved");
  });

  it("drops a step with fewer than five members", () => {
    const counts = ensembleMeans(tracks, "baseline").map((point) => point.count);
    expect(Math.min(...counts)).toBeGreaterThanOrEqual(5);
  });
});
