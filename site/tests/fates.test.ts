import { describe, expect, it } from "vitest";

import {
  FATE_WORDS,
  fateAt,
  fatesLine,
  onLand,
  positionAt,
  splitStep,
  storms,
  tally,
  tallyText,
} from "../src/fates";
import type { StepStat } from "../src/steps";
import type { TrackSample } from "../src/tracks";

const sample = (run: string, member: number, step: number, lon: number): TrackSample =>
  ({ run, member, step, lat: 30, lon, mslp_pa: 97_000 + step * 100 }) as TrackSample;

// Land is everything west of 75°W in these tracks.
const westIsLand = (lon: number): boolean => lon < -75;

describe("onLand", () => {
  it("puts Pennsylvania on land and the open Atlantic at sea", () => {
    expect(onLand(-77, 40.5)).toBe(true);
    expect(onLand(-60, 35)).toBe(false);
  });
});

describe("fates", () => {
  const all = storms(
    [
      sample("fork", 0, 1, -70),
      sample("fork", 0, 0, -72),
      sample("baseline", 1, 0, -72),
      sample("baseline", 1, 1, -74),
      sample("baseline", 1, 2, -76),
      sample("baseline", 1, 3, -77),
      sample("control", 0, 0, -72),
      sample("baseline", 0, 0, -72),
      sample("baseline", 0, 1, -71),
    ],
    westIsLand,
  );

  it("orders the ledger by run, then member, with fixes in step order", () => {
    expect(all.map((storm) => `${storm.run}${String(storm.member)}`)).toEqual([
      "baseline0",
      "baseline1",
      "fork0",
      "control0",
    ]);
    expect(all[2]?.fixes.map((fix) => fix.step)).toEqual([0, 1]);
    expect(all[2]?.fixes[1]?.hpa).toBe(971);
  });

  it("seals landfall at the first fix over land and keeps it after the storm dies", () => {
    const storm = all[1];
    if (storm === undefined) throw new Error("missing storm");
    expect(fateAt(storm, 1).kind).toBe("at-sea");
    expect(fateAt(storm, 2)).toMatchObject({ kind: "ashore", sealedAt: 2, ended: false });
    expect(fateAt(storm, 4)).toMatchObject({ kind: "ashore", sealedAt: 2, ended: true });
    expect(fateAt(storm, 4).seen?.step).toBe(3);
  });

  it("calls a storm lost at sea when the tracker loses it offshore", () => {
    const storm = all[0];
    if (storm === undefined) throw new Error("missing storm");
    expect(fateAt(storm, 1).kind).toBe("at-sea");
    expect(fateAt(storm, 2)).toMatchObject({ kind: "lost", sealedAt: 1, ended: true });
  });

  it("keeps one word per fate, even on the last step", () => {
    const storm = all[2];
    if (storm === undefined) throw new Error("missing storm");
    expect(fateAt(storm, 1).kind).toBe("at-sea");
    expect(Object.values(FATE_WORDS)).toEqual(["At sea", "Ashore", "Lost at sea"]);
  });

  it("tallies every member into exactly one fate at every step", () => {
    const baseline = all.filter((storm) => storm.run === "baseline");
    for (let step = 0; step <= 4; step += 1) {
      const counts = tally(baseline, step);
      expect(counts["at-sea"] + counts.ashore + counts.lost).toBe(baseline.length);
    }
    expect(tally(baseline, 1)).toEqual({ "at-sea": 2, ashore: 0, lost: 0 });
    expect(tallyText(tally(baseline, 4))).toBe("at sea 0 · ashore 1 · lost 1");
  });

  it("puts the split at the first holding step with more unedited storms ashore", () => {
    const baseline = all.filter((storm) => storm.run === "baseline");
    const fork = all.filter((storm) => storm.run === "fork");
    const stat = (step: number, pairedP: number | null): StepStat => ({
      step,
      validTime: "",
      forkKm: 300,
      controlKm: 10,
      pairedMembers: 8,
      pairedP,
    });
    const holding = [0, 1, 2, 3].map((step) => stat(step, 0.004));
    expect(splitStep(holding, baseline, fork)).toBe(2);
    expect(splitStep([stat(0, 0.004), stat(2, 0.5), stat(3, 0.004)], baseline, fork)).toBe(3);
    expect(splitStep(holding.slice(0, 2), baseline, fork)).toBeNull();
  });

  it("counts the verdict's fates from the ledger's own tallies", () => {
    const baseline = all.filter((storm) => storm.run === "baseline");
    const fork = all.filter((storm) => storm.run === "fork");
    for (let step = 0; step <= 4; step += 1) {
      const heads = [baseline, fork].map(
        (run) => /ashore (\d+)/.exec(tallyText(tally(run, step)))?.[1],
      );
      expect(fatesLine(baseline, fork, step)).toBe(
        `${heads[0] ?? ""} unedited ashore, ${heads[1] ?? ""} moved`,
      );
    }
    expect(fatesLine(baseline, fork, 4)).toBe("1 unedited ashore, 0 moved");
  });

  it("interpolates a position only between consecutive fixes", () => {
    const storm = all[1];
    if (storm === undefined) throw new Error("missing storm");
    expect(positionAt(storm, 1.5)?.lon).toBeCloseTo(-75);
    expect(positionAt(storm, 3.5)).toBeNull();
  });
});
