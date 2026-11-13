import { ATLANTIC_COAST } from "./atlanticCoast";
import { verdict, type StepStat } from "./steps";
import type { TrackSample } from "./tracks";

/** One six-hour fix of one member's storm. */
export interface Fix {
  readonly step: number;
  readonly lat: number;
  readonly lon: number;
  /** Central pressure in hPa, when the exporter wrote one. */
  readonly hpa: number | null;
}

/** One member of one run, followed from the start to its last fix. */
export interface Storm {
  readonly run: string;
  readonly member: number;
  readonly fixes: readonly Fix[];
  /** First fix whose centre lies over land, if the storm ever got there. */
  readonly landfall: Fix | null;
}

export type FateKind = "at-sea" | "ashore" | "lost";

/** The one word the ledger writes for each fate, whatever the step. */
export const FATE_WORDS: Readonly<Record<FateKind, string>> = {
  "at-sea": "At sea",
  ashore: "Ashore",
  lost: "Lost at sea",
};

export interface Fate {
  readonly kind: FateKind;
  /** Step the fate was sealed at: landfall for "ashore", the last fix for "lost". */
  readonly sealedAt: number | null;
  /** Last fix at or before the step: where the storm was last seen. */
  readonly seen: Fix | null;
  /** The tracker has lost the storm by this step. */
  readonly ended: boolean;
}

/** Ledger order: the run as the model ran it, the edited run, then the control. */
export const RUN_ORDER = ["baseline", "fork", "control"] as const;

/** Even-odd test against Natural Earth land rings, with lakes cut out as water. */
export function onLand(
  lon: number,
  lat: number,
  rings: readonly (readonly number[])[] = ATLANTIC_COAST,
): boolean {
  let inside = false;
  for (const ring of rings) {
    for (let i = 0, j = ring.length - 2; i < ring.length; j = i, i += 2) {
      const xi = ring[i] ?? 0;
      const yi = ring[i + 1] ?? 0;
      const xj = ring[j] ?? 0;
      const yj = ring[j + 1] ?? 0;
      if (yi > lat !== yj > lat && lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) {
        inside = !inside;
      }
    }
  }
  return inside;
}

function pressureOf(sample: TrackSample): number | null {
  const pa = (sample as { readonly mslp_pa?: unknown }).mslp_pa;
  return typeof pa === "number" && Number.isFinite(pa) ? pa / 100 : null;
}

/** Group track samples into one storm per run and member, in ledger order. */
export function storms(
  samples: readonly TrackSample[],
  land: (lon: number, lat: number) => boolean = onLand,
): readonly Storm[] {
  const byKey = new Map<string, { run: string; member: number; fixes: Fix[] }>();
  for (const sample of samples) {
    const key = `${sample.run}:${String(sample.member)}`;
    const entry = byKey.get(key) ?? { run: sample.run, member: sample.member, fixes: [] };
    entry.fixes.push({
      step: sample.step,
      lat: sample.lat,
      lon: sample.lon,
      hpa: pressureOf(sample),
    });
    byKey.set(key, entry);
  }
  const rank = (run: string): number => {
    const index = RUN_ORDER.indexOf(run as (typeof RUN_ORDER)[number]);
    return index === -1 ? RUN_ORDER.length : index;
  };
  return [...byKey.values()]
    .map(({ run, member, fixes }) => {
      const sorted = fixes.sort((a, b) => a.step - b.step);
      return {
        run,
        member,
        fixes: sorted,
        landfall: sorted.find((fix) => land(fix.lon, fix.lat)) ?? null,
      };
    })
    .sort((a, b) => rank(a.run) - rank(b.run) || a.member - b.member);
}

/** What the ledger writes against a storm at a step. */
export function fateAt(storm: Storm, step: number): Fate {
  const seen = storm.fixes.findLast((fix) => fix.step <= step) ?? null;
  const last = storm.fixes.at(-1);
  const ended = last === undefined || last.step < step;
  if (storm.landfall !== null && storm.landfall.step <= step) {
    return { kind: "ashore", sealedAt: storm.landfall.step, seen, ended };
  }
  if (ended) {
    return { kind: "lost", sealedAt: last?.step ?? null, seen, ended };
  }
  return { kind: "at-sea", sealedAt: null, seen, ended };
}

/** How one run's members stand at a step; the three fates always add up to the members. */
export type Tally = Readonly<Record<FateKind, number>>;

export function tally(members: readonly Storm[], step: number): Tally {
  const fates = members.map((storm) => fateAt(storm, step).kind);
  const count = (kind: FateKind): number => fates.filter((fate) => fate === kind).length;
  return { "at-sea": count("at-sea"), ashore: count("ashore"), lost: count("lost") };
}

/**
 * The first step at which the fork is in the data: the paired test holds and more unedited
 * storms are ashore than moved ones. Before it, a holding test shows only the edit itself.
 */
export function splitStep(
  stats: readonly StepStat[],
  unedited: readonly Storm[],
  moved: readonly Storm[],
): number | null {
  const split = stats.find(
    (stat) =>
      verdict(stat) === "separated" &&
      tally(unedited, stat.step).ashore > tally(moved, stat.step).ashore,
  );
  return split?.step ?? null;
}

/**
 * The verdict's count of fates, read from the same tallies as the ledger heads at the same
 * step, so the sentence and the table can never disagree.
 */
export function fatesLine(
  unedited: readonly Storm[],
  moved: readonly Storm[],
  step: number,
): string {
  return `${String(tally(unedited, step).ashore)} unedited ashore, ${String(tally(moved, step).ashore)} moved`;
}

/** A run's head in the ledger: one count per fate word, in the order a storm meets them. */
export function tallyText(counts: Tally): string {
  const heads: Readonly<Record<FateKind, string>> = {
    "at-sea": "at sea",
    ashore: "ashore",
    lost: "lost",
  };
  return (["at-sea", "ashore", "lost"] as const)
    .map((kind) => `${heads[kind]} ${String(counts[kind])}`)
    .join(" · ");
}

/** Position of a storm at a fractional step, between its two fixes; null outside its life. */
export function positionAt(storm: Storm, step: number): { lat: number; lon: number } | null {
  const before = storm.fixes.findLast((fix) => fix.step <= step);
  if (before === undefined) {
    return null;
  }
  if (before.step === step) {
    return before;
  }
  const after = storm.fixes.find((fix) => fix.step > step);
  if (after === undefined || after.step !== before.step + 1) {
    return null;
  }
  const t = step - before.step;
  return {
    lat: before.lat + (after.lat - before.lat) * t,
    lon: before.lon + (after.lon - before.lon) * t,
  };
}
