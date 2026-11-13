import type { TrackSample } from "./tracks";

/** Published per-step evidence for one experiment, as the exporter wrote it. */
export interface StepStat {
  readonly step: number;
  readonly validTime: string;
  readonly forkKm: number | null;
  readonly controlKm: number | null;
  readonly pairedMembers: number;
  readonly pairedP: number | null;
}

/** The exporter's matched-member rule: a mean needs a majority of an eight-member ensemble. */
export const MIN_MEAN_MEMBERS = 5;

/** Matched members the paired test needs before it runs. */
export const MIN_PAIRS = 5;

/** Level the one-sided paired test is read against. */
export const SIGNIFICANCE_LEVEL = 0.05;

export type Verdict = "separated" | "within-noise" | "too-few-pairs";

export interface MeanPoint {
  readonly step: number;
  readonly lat: number;
  readonly lon: number;
  readonly count: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nonNegative(value: unknown, what: string): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new Error(`divergence.json has an invalid ${what}`);
  }
  return value;
}

/** Read the separations and the paired test for each step of a divergence report. */
export function parseSteps(value: unknown): readonly StepStat[] {
  if (!isRecord(value) || !Array.isArray(value["steps"])) {
    throw new Error("divergence.json is not a divergence report");
  }
  return value["steps"].map((item: unknown) => {
    if (!isRecord(item)) {
      throw new Error("divergence.json contains a malformed step");
    }
    const { step, valid_time: validTime, paired_members: members } = item;
    if (typeof step !== "number" || !Number.isInteger(step) || step < 0) {
      throw new Error("divergence.json contains a malformed step");
    }
    const p = nonNegative(item["paired_p"], "paired_p");
    if (p !== null && p > 1) {
      throw new Error("divergence.json has an invalid paired_p");
    }
    return {
      step,
      validTime: typeof validTime === "string" ? validTime : "",
      forkKm: nonNegative(item["fork_sep_km"], "fork separation"),
      controlKm: nonNegative(item["control_sep_km"], "control separation"),
      pairedMembers: typeof members === "number" && Number.isInteger(members) ? members : 0,
      pairedP: p,
    };
  });
}

/** Whether the fork is farther from the baseline than the control is, at this step. */
export function verdict(stat: StepStat): Verdict {
  if (stat.pairedP === null) {
    return "too-few-pairs";
  }
  return stat.pairedP <= SIGNIFICANCE_LEVEL ? "separated" : "within-noise";
}

/**
 * Ensemble mean position per step, in step order.
 *
 * Latitude is the arithmetic mean and longitude the circular mean, as in the
 * exporter, and a step with fewer members than the majority rule is dropped.
 */
export function ensembleMeans(
  samples: readonly TrackSample[],
  run: string,
  minMembers = MIN_MEAN_MEMBERS,
): readonly MeanPoint[] {
  const byStep = new Map<number, TrackSample[]>();
  for (const sample of samples) {
    if (sample.run === run) {
      byStep.set(sample.step, [...(byStep.get(sample.step) ?? []), sample]);
    }
  }
  const means: MeanPoint[] = [];
  for (const [step, members] of [...byStep].sort((a, b) => a[0] - b[0])) {
    if (members.length < minMembers) {
      continue;
    }
    const radians = members.map((member) => (member.lon * Math.PI) / 180);
    const sin = members.reduce((sum, _member, index) => sum + Math.sin(radians[index] ?? 0), 0);
    const cos = members.reduce((sum, _member, index) => sum + Math.cos(radians[index] ?? 0), 0);
    means.push({
      step,
      lat: members.reduce((sum, member) => sum + member.lat, 0) / members.length,
      lon: (Math.atan2(sin, cos) * 180) / Math.PI,
      count: members.length,
    });
  }
  return means;
}

/** The last step the baseline's ensemble mean exists for: where the story ends. */
export function lastMeanStep(samples: readonly TrackSample[]): number {
  return Math.max(0, ...ensembleMeans(samples, "baseline").map((point) => point.step));
}

const KM_PER_DEGREE = 111.195;

/** Great-circle distance between two points, on the exporter's sphere. */
export function greatCircleKm(aLat: number, aLon: number, bLat: number, bLon: number): number {
  const toRad = Math.PI / 180;
  const dLat = (bLat - aLat) * toRad;
  const dLon = (bLon - aLon) * toRad;
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(aLat * toRad) * Math.cos(bLat * toRad) * Math.sin(dLon / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.min(1, Math.sqrt(h)));
}

export function kmToDegrees(km: number): number {
  return km / KM_PER_DEGREE;
}

export function formatP(p: number): string {
  if (p < 0.001) {
    return "< 0.001";
  }
  return p < 0.01 ? p.toFixed(4) : p.toFixed(3);
}

/** Kilometres as every page prints them: whole numbers with thousands separators. */
export function formatKm(value: number | null): string {
  return value === null ? "none" : Math.round(value).toLocaleString("en-GB");
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** Time of a step as every page labels it: day "27 Oct" and hour "12:00 UTC". */
export function stepTime(initTime: string, step: number): { day: string; hour: string } {
  const start = Date.parse(initTime.endsWith("Z") ? initTime : `${initTime}Z`);
  const at = new Date(start + step * 6 * 3_600_000);
  return {
    day: `${String(at.getUTCDate())} ${MONTHS[at.getUTCMonth()] ?? ""}`,
    hour: `${String(at.getUTCHours()).padStart(2, "0")}:00 UTC`,
  };
}
