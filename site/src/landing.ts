import { element as pick } from "./dom";
import { FRAME_COUNT } from "./atlas";
import {
  FATE_WORDS,
  fateAt,
  fatesLine,
  positionAt,
  RUN_ORDER,
  splitStep,
  storms,
  tally,
  tallyText,
  type Fate,
  type Storm,
} from "./fates";
import { ATLAS_BASE_URL } from "./loadExperiment";
import { isManifest, type Manifest } from "./manifest";
import { decodeAtlas } from "./pressure";
import type { Relief, ReliefRun } from "./relief";
import { drawSeparation, type DrawnChart } from "./separationChart";
import {
  ensembleMeans,
  formatKm,
  formatP,
  MIN_PAIRS,
  SIGNIFICANCE_LEVEL,
  parseSteps,
  stepTime,
  verdict,
  type StepStat,
} from "./steps";
import { parseTracks, type TrackSample } from "./tracks";

export interface LandingOptions {
  readonly id: string;
  readonly reducedMotion: boolean;
  /** The social card: one still frame, no controls, no motion. */
  readonly card: boolean;
}

export interface Landing {
  readonly dispose: () => void;
}

/** Six-hour steps the watch turns through per second while the forecast plays. */
const STEPS_PER_SECOND = 2.6;
/** Through the landfall and its aftermath the watch slows, so the turn reads as one event. */
const LANDFALL_STEPS_PER_SECOND = 1.5;
/** How long one ledger row takes to flash; rows that seal together flash one after another. */
const FLASH_MS = 300;

const RUN_LABEL: Readonly<Record<string, string>> = {
  baseline: "As the model ran it",
  fork: "Moved {edit} km east",
  control: "Control, new seeds",
};

const SMALL = [
  "zero",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
  "eleven",
  "twelve",
  "thirteen",
  "fourteen",
  "fifteen",
  "sixteen",
  "seventeen",
  "eighteen",
  "nineteen",
];
const TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

/** A count under a hundred, in words, as a ledger would write it. */
export function inWords(n: number): string {
  if (n < 20) return SMALL[n] ?? String(n);
  if (n >= 100) return String(n);
  const tens = TENS[Math.floor(n / 10)] ?? "";
  return n % 10 === 0 ? tens : `${tens}-${SMALL[n % 10] ?? ""}`;
}

async function fetchJson(url: string): Promise<unknown> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} responded ${String(response.status)}`);
  }
  return response.json();
}

function fillText(values: Readonly<Record<string, string>>): void {
  for (const target of document.querySelectorAll<HTMLElement>("[data-fill]")) {
    const value = values[target.dataset["fill"] ?? ""];
    if (value !== undefined) {
      target.textContent = value;
    }
  }
}

/** Fill the numbers that belong to the method, not to a run, wherever the pages quote them. */
export function fillConstants(): void {
  fillText({
    level: String(SIGNIFICANCE_LEVEL),
    minPairs: String(MIN_PAIRS),
    frames: String(FRAME_COUNT),
  });
}

function describeRun(manifest: Manifest, steps: readonly StepStat[], count: number) {
  const initTime = manifest.init?.time ?? "";
  const start = stepTime(initTime, 0);
  const tested = steps.filter((stat) => stat.pairedP !== null);
  const separated = steps.filter((stat) => verdict(stat) === "separated");
  const peak = Math.max(0, ...steps.map((stat) => stat.forkKm ?? 0));
  return {
    start: `${start.day} ${initTime.slice(0, 4)}, ${start.hour}`,
    members: String(manifest.members ?? 0),
    storms: inWords(count),
    model: manifest.model?.name ?? "",
    modelName: (manifest.model?.name ?? "").split("_<")[0]?.replaceAll("_", " ") ?? "",
    edit: String(manifest.perturbation?.params["east_km"] ?? ""),
    seedBaseline: String(manifest.seeds?.baseline ?? ""),
    seedControl: String(manifest.seeds?.control ?? ""),
    tested: String(tested.length),
    separated: String(separated.length),
    peak: formatKm(peak),
    checkpoint: manifest.model?.checkpoint ?? "",
    md5: manifest.model?.md5 ?? "",
    device: manifest.runtime?.device ?? "",
    created: manifest.created?.slice(0, 10) ?? "",
  };
}

function renderStepsTable(steps: readonly StepStat[], initTime: string, lastStep: number): void {
  const body = pick("#steps-table tbody", HTMLTableSectionElement);
  body.replaceChildren(
    ...steps
      .filter((stat) => stat.step <= lastStep)
      .map((stat) => {
        const when = stepTime(initTime, stat.step);
        const row = document.createElement("tr");
        for (const text of [
          String(stat.step),
          `${when.day}, ${when.hour}`,
          formatKm(stat.forkKm),
          formatKm(stat.controlKm),
          String(stat.pairedMembers),
          stat.pairedP === null ? "n/a" : formatP(stat.pairedP),
        ]) {
          const cell = document.createElement("td");
          cell.textContent = text;
          row.append(cell);
        }
        return row;
      }),
  );
}

function renderLimits(limits: readonly string[]): void {
  pick("#limits", HTMLOListElement).replaceChildren(
    ...limits.map((limit) => {
      const item = document.createElement("li");
      item.textContent = limit;
      return item;
    }),
  );
}

const degrees = (value: number, positive: string, negative: string): string =>
  `${Math.abs(value).toFixed(1)}°${value >= 0 ? positive : negative}`;

interface Row {
  readonly storm: Storm;
  readonly row: HTMLTableRowElement;
  readonly button: HTMLButtonElement;
  readonly label: string;
  readonly fate: HTMLTableCellElement;
  readonly seen: HTMLTableCellElement;
  readonly hpa: HTMLTableCellElement;
  kind: Fate["kind"] | null;
}

/** The dial of the watch: one tick per six-hour step, the date every fourth tick. */
function drawDial(dial: SVGSVGElement, initTime: string, lastStep: number, tested: number) {
  const ns = "http://www.w3.org/2000/svg";
  const add = (tag: string, attributes: Record<string, string | number>): SVGElement => {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
    dial.append(node);
    return node;
  };
  // Eleven days round the face: the forecast's ten and one left blank, so its last day
  // never lands on its first.
  const turns = lastStep + 4;
  const angle = (step: number): number => (step / turns) * Math.PI * 2 - Math.PI / 2;
  dial.replaceChildren();
  add("circle", { r: 66, class: "dial-face" });
  add("circle", { r: 60, class: "dial-rim" });
  for (let step = 0; step <= lastStep; step += 1) {
    const a = angle(step);
    const major = step % 4 === 0;
    const inner = major ? 50 : 55;
    add("line", {
      x1: Math.cos(a) * inner,
      y1: Math.sin(a) * inner,
      x2: Math.cos(a) * 60,
      y2: Math.sin(a) * 60,
      class: major ? "dial-tick major" : "dial-tick",
    });
    if (major) {
      const label = add("text", { x: Math.cos(a) * 40, y: Math.sin(a) * 40 + 3.5 });
      label.textContent = stepTime(initTime, step).day.split(" ")[0] ?? "";
    }
  }
  // The arc of steps the paired test could run at: the watch's tested quadrant.
  const r = 64;
  const end = angle(tested);
  add("path", {
    d: `M0,${String(-r)} A${String(r)},${String(r)} 0 ${end - angle(0) > Math.PI ? "1" : "0"} 1 ${String(Math.cos(end) * r)},${String(Math.sin(end) * r)}`,
    class: "dial-tested",
  });
  const hand = add("line", { x1: 0, y1: 0, x2: 0, y2: -46, class: "dial-hand" });
  add("circle", { r: 4, class: "dial-pin" });
  return (step: number): void => {
    const a = angle(step);
    hand.setAttribute("x2", String(Math.cos(a) * 46));
    hand.setAttribute("y2", String(Math.sin(a) * 46));
  };
}

/**
 * Open the fork page: the relief of one run's pressure, every member's fate in the ledger,
 * and the paired test's stamp, all tied to one step that the watch turns.
 */
export async function openLanding(options: LandingOptions): Promise<Landing> {
  const hero = pick("#hero", HTMLElement);
  const scene = pick("#scene", HTMLElement);
  const canvas = pick("#relief", HTMLCanvasElement);
  const still = pick("#relief-still", HTMLImageElement);
  const failure = pick("#plate-error", HTMLElement);
  const dial = pick("#dial", SVGSVGElement);
  const play = pick("#landing-play", HTMLButtonElement);
  const runButtons = {
    baseline: pick("#run-baseline", HTMLButtonElement),
    fork: pick("#run-fork", HTMLButtonElement),
  } as const;
  const tags = pick("#tags", HTMLElement);
  const plate = pick(".watch", HTMLElement);
  const bigSvg = pick("#separation-chart", SVGSVGElement);

  const base = `${ATLAS_BASE_URL}${options.id}/`;
  hero.dataset["state"] = "loading";
  failure.hidden = true;

  let manifest: Manifest;
  let steps: readonly StepStat[];
  let samples: readonly TrackSample[];
  try {
    const rawManifest = await fetchJson(`${base}manifest.json`);
    if (!isManifest(rawManifest)) {
      throw new Error(`${base}manifest.json is not a stormfork manifest`);
    }
    manifest = rawManifest;
    const [rawTracks, rawSteps] = await Promise.all([
      fetchJson(`${base}tracks.json`),
      fetchJson(`${base}divergence.json`),
    ]);
    steps = parseSteps(rawSteps);
    samples = parseTracks(rawTracks);
  } catch (error) {
    hero.dataset["state"] = "error";
    failure.hidden = false;
    pick("#plate-error-detail", HTMLElement).textContent =
      error instanceof Error ? error.message : "unknown error";
    return { dispose: () => undefined };
  }

  const all = storms(samples);
  const membersOf = (run: string): readonly Storm[] => all.filter((storm) => storm.run === run);
  const initTime = manifest.init?.time ?? "";
  const lastStep = Math.min(manifest.steps ?? FRAME_COUNT - 1, FRAME_COUNT - 1);
  const editKm = manifest.perturbation?.params["east_km"] ?? 0;
  const edit = String(editKm);
  const stepAt = new Map(steps.map((stat) => [stat.step, stat] as const));
  const chartLast = Math.max(0, ...steps.filter((s) => s.forkKm !== null).map((s) => s.step));
  const lastTested = Math.max(0, ...steps.filter((s) => s.pairedP !== null).map((s) => s.step));
  const restStep = Math.max(
    0,
    ...steps.filter((s) => verdict(s) === "separated").map((s) => s.step),
  );
  // The first unedited landfall. The heads cross the coast on the unedited relief, and one
  // six-hour step later the page turns to the moved run.
  const landfallStep = Math.min(
    ...all.filter((s) => s.run === "baseline" && s.landfall).map((s) => s.landfall?.step ?? 0),
  );
  const turnStep = landfallStep + 1;
  // Until the fates split, a holding test shows only the edit itself; the stamp inks then.
  const split = splitStep(steps, membersOf("baseline"), membersOf("fork")) ?? Infinity;

  fillText(describeRun(manifest, steps, all.length));
  for (const button of [runButtons.fork]) button.textContent = `Moved ${edit} km east`;
  renderLimits(manifest.limits ?? []);
  renderStepsTable(steps, initTime, chartLast);

  // The ledger: one row per storm, grouped by run.
  const body = pick("#fates", HTMLTableSectionElement);
  const rows: Row[] = [];
  const groupCounts = new Map<string, HTMLElement>();
  body.replaceChildren();
  for (const run of RUN_ORDER) {
    const members = all.filter((storm) => storm.run === run);
    if (members.length === 0) continue;
    const head = body.insertRow();
    head.className = "group";
    const cell = head.insertCell();
    cell.colSpan = 4;
    const name = document.createElement("span");
    name.textContent = (RUN_LABEL[run] ?? run).replace("{edit}", edit);
    const count = document.createElement("span");
    count.className = "group-count";
    cell.append(name, count);
    groupCounts.set(run, count);
    for (const storm of members) {
      const row = body.insertRow();
      row.dataset["run"] = run;
      const who = row.insertCell();
      const button = document.createElement("button");
      button.type = "button";
      button.className = "who";
      button.textContent = `No. ${String(storm.member)}`;
      const label = `${name.textContent}, member ${String(storm.member)}`;
      button.setAttribute("aria-label", `${label}: burn its track`);
      who.append(button);
      rows.push({
        storm,
        row,
        button,
        label,
        fate: row.insertCell(),
        seen: row.insertCell(),
        hpa: row.insertCell(),
        kind: null,
      });
    }
  }

  const setHand = drawDial(dial, initTime, lastStep, lastTested);
  dial.setAttribute("aria-valuemax", String(lastStep));

  let step = options.reducedMotion || options.card ? restStep : 0;
  let run: ReliefRun = options.reducedMotion || options.card ? "fork" : "baseline";
  let focus: Storm | null = null;
  let playing = false;
  let turned = step >= turnStep;
  let shownWhole = -1;
  let relief: Relief | null = null;
  let flashFree = 0;
  let stampInked = false;
  const motion = !options.reducedMotion && !options.card;
  // The unedited storms make landfall at the turn and are lost inland within a day or two.
  const aftermath = Math.max(
    landfallStep,
    ...all.filter((s) => s.run === "baseline").map((s) => s.fixes.at(-1)?.step ?? 0),
  );

  const palette = getComputedStyle(document.documentElement);
  const inks = {
    ground: palette.getPropertyValue("--ground").trim(),
    ink: palette.getPropertyValue("--ink").trim(),
  };

  const short = (at: number): string => {
    const when = stepTime(initTime, at);
    return `${when.day} ${when.hour.replace(" UTC", "")}`;
  };

  function ledger(whole: number, animate: boolean): void {
    for (const entry of rows) {
      const fate = fateAt(entry.storm, whole);
      const word = document.createElement("span");
      word.className = "fate-word";
      word.textContent = FATE_WORDS[fate.kind];
      const when = document.createElement("span");
      when.className = "fate-when";
      when.textContent = fate.sealedAt === null ? "" : short(fate.sealedAt);
      entry.fate.replaceChildren(word, " ", when);
      entry.seen.textContent =
        fate.seen === null
          ? ""
          : `${degrees(fate.seen.lat, "N", "S")} ${degrees(fate.seen.lon, "E", "W")}`;
      entry.hpa.textContent = fate.seen?.hpa == null ? "" : String(Math.round(fate.seen.hpa));
      entry.row.classList.toggle("gone", fate.ended);
      entry.button.setAttribute(
        "aria-label",
        `${entry.label}${fate.ended ? ", no longer tracked" : ""}: burn its track`,
      );
      const onRelief = entry.storm.run !== "control";
      if (
        animate &&
        onRelief &&
        entry.kind !== null &&
        entry.kind !== fate.kind &&
        fate.kind !== "at-sea"
      ) {
        // The fate of a storm on the relief is inked: its fate cell flashes in the light ink,
        // then settles. Cells sealed in the same instant queue, so only one is ever lit.
        const now = performance.now();
        const delay = Math.max(0, flashFree - now);
        flashFree = now + delay + FLASH_MS;
        const lit = { background: inks.ink, color: inks.ground, easing: "steps(1, end)" };
        entry.fate.animate(
          [lit, { ...lit, offset: 0.7 }, { background: "transparent", color: inks.ink }],
          { duration: FLASH_MS, delay },
        );
      }
      entry.kind = fate.kind;
    }
    for (const [groupRun, element] of groupCounts) {
      element.textContent = tallyText(tally(membersOf(groupRun), whole));
    }
  }

  function verdictAt(whole: number): void {
    // Once the test closes, the stamp keeps its last verdict; only its border says it is over.
    const closed = whole > lastTested;
    const at = closed ? lastTested : whole;
    const stat = stepAt.get(at);
    const stamp = pick("#stamp", HTMLElement);
    const said = pick("#verdict", HTMLElement);
    const figures = pick("#figures", HTMLElement);
    const outcome = stat === undefined ? "too-few-pairs" : verdict(stat);
    const pairs = stat?.pairedMembers ?? 0;
    const early = outcome === "separated" && at < split;
    const inked = outcome === "separated" && !early;
    if (inked && !stampInked && motion && playing) {
      stamp.animate(
        [
          { transform: "rotate(-3deg) scale(1.7)", opacity: 0 },
          { transform: "rotate(-3deg) scale(1)", opacity: 1 },
        ],
        { duration: 240, easing: "cubic-bezier(0.3, 0, 0.9, 0.6)" },
      );
    }
    stampInked = inked;
    stamp.dataset["verdict"] = outcome;
    stamp.toggleAttribute("data-early", early);
    stamp.toggleAttribute("data-closed", closed);
    const held = stepAt.get(whole)?.pairedMembers ?? 0;
    pick("#pairs-now", HTMLElement).textContent = closed
      ? `test closed at ${String(pairs)}\u00a0pairs`
      : `test pairs now\u00a0${String(held)}`;
    const control = formatKm(stat?.controlKm ?? null);
    // The edit itself puts the storms 300 km apart, so what counts is how far past it they grew.
    const grown = stat?.forkKm == null ? null : stat.forkKm - editKm;
    const apart =
      grown === null
        ? ""
        : `Moved ${grown < 0 ? "\u2212" : ""}${formatKm(Math.abs(grown))} km past the ${edit} km edit; control ${control} km. `;
    if (stat?.pairedP == null) {
      stamp.replaceChildren("Unconfirmed");
      said.textContent = `Fewer than ${String(MIN_PAIRS)} pairs still hold a storm, so the test does not run.`;
      figures.textContent = `${apart}${String(pairs)} ${pairs === 1 ? "pair" : "pairs"} left.`;
      return;
    }
    // An exact signed-rank p of 1 / 2^n means every one of the n pairs points the same way.
    const unanimous = Math.abs(stat.pairedP * 2 ** pairs - 1) < 1e-9;
    const share = unanimous ? `${String(pairs)} of ${String(pairs)} ` : "";
    const word =
      outcome === "separated" ? `${share}${early ? "apart" : "confirmed"}` : "Not confirmed";
    const note = document.createElement("small");
    note.textContent = `Test closed at ${String(pairs)} pairs`;
    stamp.replaceChildren(word, ...(closed ? [note] : []));
    const farther = `${unanimous ? "In all" : "Across"} ${String(pairs)} pairs the moved storm sat farther from its twin than the control`;
    if (early) {
      said.textContent = `${farther}; so far that is the ${edit} km edit, not a split.`;
    } else if (outcome === "separated") {
      // Counted at the step on show, the ledger's step, even after the test has closed.
      const fates = fatesLine(membersOf("baseline"), membersOf("fork"), whole);
      said.textContent = closed
        ? `${farther}. By ${short(whole)}: ${fates}.`
        : `${farther}, and the fates split: ${fates}.`;
    } else {
      said.textContent = `Across ${String(pairs)} pairs, the gap is inside what the control shows the model does on its own.`;
    }
    figures.textContent = `${apart}Signed-rank p\u00a0${formatP(stat.pairedP)}${unanimous ? `, the floor for ${String(pairs)} pairs` : ""}.`;
  }

  // The paired test's row: one cell per step, the present one marked.
  const cells = pick("#cells", HTMLOListElement);
  cells.replaceChildren(
    ...Array.from({ length: lastStep + 1 }, (_, at) => {
      const item = document.createElement("li");
      const stat = stepAt.get(at);
      item.dataset["verdict"] = stat === undefined ? "too-few-pairs" : verdict(stat);
      return item;
    }),
  );

  /**
   * Name the two runs' mean storms on the relief. Each name sits at the end of a leader
   * drawn clear of the storm: the unedited run's toward the land, the moved run's out to sea.
   */
  function placeTags(whole: number): void {
    if (relief === null) {
      tags.replaceChildren();
      return;
    }
    const width = scene.clientWidth;
    const height = scene.clientHeight;
    const reach = (width < 600 ? 80 : 120) * 0.7;
    const sceneBox = scene.getBoundingClientRect();
    const plateBox = plate.getBoundingClientRect();
    // The watch's plate, in the scene's own coordinates, with a margin.
    const kept = {
      left: plateBox.left - sceneBox.left - 8,
      right: plateBox.right - sceneBox.left + 8,
      top: plateBox.top - sceneBox.top - 8,
      bottom: plateBox.bottom - sceneBox.top + 8,
    };
    type Box = { left: number; right: number; top: number; bottom: number };
    const taken: Box[] = [kept];
    const leaders: [number, number, number, number][] = [];
    const apart = (a: Box, b: Box): boolean =>
      a.right < b.left || a.left > b.right || a.bottom < b.top || a.top > b.bottom;
    // Two leaders cross when each one's ends lie on opposite sides of the other.
    const side = (ax: number, ay: number, bx: number, by: number, px: number, py: number) =>
      Math.sign((bx - ax) * (py - ay) - (by - ay) * (px - ax));
    const crosses = (x1: number, y1: number, x2: number, y2: number): boolean =>
      leaders.some(
        ([a, b, c, d]) =>
          side(a, b, c, d, x1, y1) !== side(a, b, c, d, x2, y2) &&
          side(x1, y1, x2, y2, a, b) !== side(x1, y1, x2, y2, c, d),
      );
    const boxAt = (x: number, y: number, dx: number, label: string): Box => {
      // Spaced capitals at 13px run about 10px a letter.
      const wide = label.length * 10 + 12;
      const left = dx < 0 ? x - wide : x;
      return { left, right: left + wide, top: y - 13, bottom: y + 13 };
    };
    const fits = (box: Box): boolean =>
      box.left >= 4 &&
      box.right <= width - 4 &&
      box.top >= 4 &&
      box.bottom <= height - 30 &&
      taken.every((other) => apart(box, other));
    const ns = "http://www.w3.org/2000/svg";
    const lines = document.createElementNS(ns, "svg");
    lines.classList.add("tag-lines");
    const labels: HTMLElement[] = [];
    const shown = (["baseline", "fork"] as const).flatMap((tagRun) => {
      const centre = ensembleMeans(samples, tagRun, 1).find((point) => point.step === whole);
      // The name points at a live head, the one nearest the run's mean, never at empty sea.
      const heads = membersOf(tagRun).flatMap((storm) => positionAt(storm, whole) ?? []);
      const gap = (p: { lon: number; lat: number }): number =>
        centre === undefined ? 0 : (p.lon - centre.lon) ** 2 + (p.lat - centre.lat) ** 2;
      const mean = centre === undefined ? undefined : heads.sort((a, b) => gap(a) - gap(b))[0];
      const at = mean === undefined ? null : relief?.project(mean.lon, mean.lat);
      if (mean === undefined || at == null) return [];
      if (at.x < 0 || at.x > width || at.y < 0 || at.y > height) return [];
      // A storm under the watch's plate is hidden, and so is its name.
      if (at.x > kept.left && at.x < kept.right && at.y > kept.top && at.y < kept.bottom) return [];
      // A leader starts outside the storm's rings, so it never cuts through them.
      // Only the run whose pressure is drawn has rings.
      const rings = tagRun === run ? (relief?.ringReach(mean.lon, mean.lat) ?? 0) : 0;
      return [{ tagRun, mean, at, clear: rings + 9 }];
    });
    // The heavy rings of the storm whose pressure is drawn carry their values, set on the ring.
    const values: HTMLElement[] = [];
    for (const { tagRun, mean } of shown) {
      if (tagRun !== run) continue;
      for (const ring of relief.ringLabels(mean.lon, mean.lat)) {
        const box = { left: ring.x - 15, right: ring.x + 15, top: ring.y - 8, bottom: ring.y + 8 };
        if (!fits(box)) continue;
        taken.push(box);
        const value = document.createElement("p");
        value.className = "ring-value";
        value.style.transform = `translate(${String(Math.round(ring.x))}px, ${String(Math.round(ring.y))}px)`;
        value.textContent = String(ring.hpa);
        values.push(value);
      }
    }
    // No name may sit on either storm.
    for (const { at, clear } of shown) {
      const r = Math.max(28, clear * 0.7);
      taken.push({ left: at.x - r, right: at.x + r, top: at.y - r, bottom: at.y + r });
    }
    for (const { tagRun, at, clear } of shown) {
      const label = tagRun === "fork" ? `Moved ${edit} km` : "Unedited";
      // The two leaders leave at different angles, so they never read as one repeated mark:
      // the unedited name climbs steeply toward the land, the moved one runs flat out to sea.
      const leans: readonly (readonly [number, number])[] =
        tagRun === "baseline"
          ? [
              [-0.5, -1],
              [-1, -0.45],
              [0.5, -1],
              [-0.5, 1],
            ]
          : [
              [1, 0.35],
              [1, -0.4],
              [0.45, 1],
              [-1, 0.35],
            ];
      const choice = [1, 1.5, 0.7]
        .flatMap((scale) =>
          leans.map(([x, y]) => ({
            dx: x,
            dy: y,
            far: Math.max(reach * scale, clear * 0.71 + 24),
          })),
        )
        .find(({ dx: x, dy: y, far }) => {
          const ex = at.x + x * far;
          const ey = at.y + y * far;
          return fits(boxAt(ex, ey, x, label)) && !crosses(at.x, at.y, ex, ey);
        });
      if (choice === undefined) continue;
      const { dx, dy, far } = choice;
      const end = { x: at.x + dx * far, y: at.y + dy * far };
      taken.push(boxAt(end.x, end.y, dx, label));
      leaders.push([at.x, at.y, end.x, end.y]);
      for (const kind of ["knock", "ink"]) {
        const line = document.createElementNS(ns, "line");
        line.classList.add(kind);
        // Anchored at the head itself; its knock-out cuts a clean path through the rings.
        line.setAttribute("x1", String(Math.round(at.x + dx * 6)));
        line.setAttribute("y1", String(Math.round(at.y + dy * 6)));
        line.setAttribute("x2", String(Math.round(end.x)));
        line.setAttribute("y2", String(Math.round(end.y)));
        lines.append(line);
      }
      const tag = document.createElement("p");
      tag.className = "tag";
      tag.dataset["side"] = dx < 0 ? "left" : "right";
      tag.style.transform = `translate(${String(Math.round(end.x))}px, ${String(Math.round(end.y))}px)`;
      tag.textContent = label;
      labels.push(tag);
    }
    // A storm that has left the map is an arrow inset from the frame's edge, pointing where it
    // went, labelled with its run and how many of that run left that way.
    const onRelief = all.filter((storm) => storm.run !== "control" || whole >= lastStep);
    const exits = new Map<
      string,
      { run: string; x: number; y: number; angle: number; count: number }
    >();
    const struck: { x: number; y: number }[] = [];
    for (const storm of onRelief) {
      const there = positionAt(storm, Math.min(whole, storm.fixes.at(-1)?.step ?? 0));
      const p = there === null ? null : relief.project(there.lon, there.lat);
      if (p == null) continue;
      if (p.x >= 0 && p.x <= width && p.y >= 0 && p.y <= height) {
        // A storm the tracker has lost keeps the ledger's cross where it was last seen.
        if (fateAt(storm, whole).ended) struck.push(p);
        continue;
      }
      const x = Math.min(Math.max(p.x, 24), width - 24);
      const y = Math.min(Math.max(p.y, 24), height - 52);
      const key = `${storm.run}:${String(Math.round(x / 96))},${String(Math.round(y / 96))}`;
      const exit = exits.get(key);
      if (exit === undefined)
        exits.set(key, { run: storm.run, x, y, angle: Math.atan2(p.y - y, p.x - x), count: 1 });
      else exit.count += 1;
    }
    for (const kind of ["knock", "ink"])
      for (const { x, y } of struck)
        for (const turn of [-1, 1]) {
          const stroke = document.createElementNS(ns, "line");
          stroke.classList.add(kind);
          stroke.setAttribute("x1", String(Math.round(x - 4)));
          stroke.setAttribute("y1", String(Math.round(y - 4 * turn)));
          stroke.setAttribute("x2", String(Math.round(x + 4)));
          stroke.setAttribute("y2", String(Math.round(y + 4 * turn)));
          lines.append(stroke);
        }
    const runWords: Readonly<Record<string, string>> = {
      baseline: "unedited",
      fork: "moved",
      control: "control",
    };
    for (const { run: exitRun, x, y, angle, count } of exits.values()) {
      const [dx, dy] = [Math.cos(angle), Math.sin(angle)];
      const tip = { x: x + dx * 8, y: y + dy * 8 };
      const strokes: [number, number, number, number][] = [
        [x - dx * 10, y - dy * 10, tip.x, tip.y],
        ...[-1, 1].map((turn): [number, number, number, number] => [
          tip.x,
          tip.y,
          tip.x - 6 * Math.cos(angle + turn * 0.6),
          tip.y - 6 * Math.sin(angle + turn * 0.6),
        ]),
      ];
      for (const kind of ["knock", "ink"])
        for (const [x1, y1, x2, y2] of strokes) {
          const stroke = document.createElementNS(ns, "line");
          stroke.classList.add(kind);
          stroke.setAttribute("x1", String(Math.round(x1)));
          stroke.setAttribute("y1", String(Math.round(y1)));
          stroke.setAttribute("x2", String(Math.round(x2)));
          stroke.setAttribute("y2", String(Math.round(y2)));
          lines.append(stroke);
        }
      // The label sits behind the arrow's tail, on the frame's side of it.
      const exit = document.createElement("p");
      exit.className = "exit-value";
      exit.dataset["side"] = Math.abs(dx) < Math.abs(dy) ? "middle" : dx > 0 ? "left" : "right";
      exit.style.transform = `translate(${String(Math.round(x - dx * 14))}px, ${String(Math.round(y - dy * 14 + (Math.abs(dx) < Math.abs(dy) ? -dy * 6 : 0)))}px)`;
      exit.textContent = `${String(count)} ${runWords[exitRun] ?? exitRun}`;
      values.push(exit);
    }
    tags.replaceChildren(lines, ...values, ...labels);
  }

  function show(at: number): void {
    step = Math.min(Math.max(at, 0), lastStep);
    const whole = Math.floor(step);
    setHand(step);
    const when = stepTime(initTime, whole);
    pick("#when", HTMLElement).textContent = `${when.day}, ${when.hour.replace(" UTC", "")}`;
    pick("#stepline", HTMLElement).textContent =
      `UTC, step ${String(whole)} of ${String(lastStep)}, +${String(whole * 6)} h`;
    dial.setAttribute("aria-valuenow", String(whole));
    dial.setAttribute("aria-valuetext", `${when.day}, ${when.hour}`);
    pick("#caption", HTMLElement).textContent =
      `Sea-level pressure of member 0, ${run === "fork" ? `moved ${edit} km east` : "as the model ran it"}; isobars every 4\u00a0hPa below 1000`;
    for (const [key, button] of Object.entries(runButtons)) {
      button.setAttribute("aria-pressed", String(key === run));
    }
    if (whole !== shownWhole) {
      ledger(whole, motion && playing);
      for (const [index, item] of [...cells.children].entries()) {
        item.classList.toggle("now", index === whole);
        item.classList.toggle("past", index < whole);
      }
      shownWhole = whole;
    }
    verdictAt(whole);
    placeTags(whole);
  }

  function setPlaying(next: boolean): void {
    playing = next;
    play.textContent = next ? "Pause" : step >= lastStep ? "Replay" : "Play";
    play.setAttribute("aria-pressed", String(next));
  }

  let visible = true;
  let request: number | null = null;
  let previous = performance.now();

  function frame(now: number): void {
    request = null;
    const dt = Math.min((now - previous) / 1000, 0.1);
    previous = now;
    if (playing) {
      const slow = step >= landfallStep - 1 && step < aftermath + 1;
      const next = step + dt * (slow ? LANDFALL_STEPS_PER_SECOND : STEPS_PER_SECOND);
      if (!turned && next >= turnStep) {
        // The signature turn: the storm as run, ashore, gives way to the moved one at sea.
        turned = true;
        run = "fork";
        relief?.dissolve(now);
      }
      show(next);
      if (step >= lastStep) setPlaying(false);
    }
    relief?.set({ step, run, focus }, now);
    relief?.render(now, dt, true);
    if (visible && motion) request = requestAnimationFrame(frame);
  }

  function redraw(): void {
    if (motion) {
      if (request === null && visible) request = requestAnimationFrame(frame);
      return;
    }
    relief?.set({ step, run, focus }, performance.now());
    relief?.render(performance.now(), 0, false);
  }

  function jump(to: number, dissolve: boolean): void {
    setPlaying(false);
    const target = Math.round(Math.min(Math.max(to, 0), lastStep));
    if (target === Math.floor(step) && Number.isInteger(step)) return;
    if (dissolve && motion) relief?.dissolve(performance.now());
    show(target);
    setPlaying(false);
    redraw();
  }

  function setRun(next: ReliefRun): void {
    if (next === run) return;
    turned = true;
    run = next;
    if (motion) relief?.dissolve(performance.now());
    show(step);
    redraw();
  }

  function setFocus(next: Storm | null): void {
    if (next === focus) return;
    focus = next;
    for (const entry of rows) entry.row.classList.toggle("focused", entry.storm === next);
    redraw();
  }

  const events = new AbortController();
  const listeners = { signal: events.signal };
  play.addEventListener(
    "click",
    () => {
      if (playing) {
        setPlaying(false);
        return;
      }
      if (step >= lastStep) {
        show(0);
        turned = false;
        if (run !== "baseline") {
          run = "baseline";
          relief?.dissolve(performance.now());
        }
      }
      setPlaying(true);
      redraw();
    },
    listeners,
  );
  runButtons.baseline.addEventListener(
    "click",
    () => {
      setRun("baseline");
    },
    listeners,
  );
  runButtons.fork.addEventListener(
    "click",
    () => {
      setRun("fork");
    },
    listeners,
  );
  dial.addEventListener(
    "keydown",
    (event) => {
      const by: Record<string, number> = {
        ArrowRight: 1,
        ArrowUp: 1,
        ArrowLeft: -1,
        ArrowDown: -1,
        PageUp: 4,
        PageDown: -4,
      };
      const delta = by[event.key];
      const whole = Math.floor(step);
      if (delta !== undefined) jump(whole + delta, true);
      else if (event.key === "Home") jump(0, true);
      else if (event.key === "End") jump(lastStep, true);
      else return;
      event.preventDefault();
    },
    listeners,
  );
  const stepFromPointer = (event: PointerEvent): number => {
    const box = dial.getBoundingClientRect();
    const a =
      Math.atan2(
        event.clientY - box.top - box.height / 2,
        event.clientX - box.left - box.width / 2,
      ) +
      Math.PI / 2;
    return Math.round(((((a / (Math.PI * 2)) % 1) + 1) % 1) * (lastStep + 1)) % (lastStep + 1);
  };
  let turning = false;
  dial.addEventListener(
    "pointerdown",
    (event) => {
      turning = true;
      dial.setPointerCapture(event.pointerId);
      jump(stepFromPointer(event), true);
    },
    listeners,
  );
  dial.addEventListener(
    "pointermove",
    (event) => {
      if (turning) jump(stepFromPointer(event), false);
    },
    listeners,
  );
  dial.addEventListener(
    "pointerup",
    () => {
      turning = false;
    },
    listeners,
  );
  for (const entry of rows) {
    const button = entry.row.querySelector("button");
    entry.row.addEventListener(
      "pointerenter",
      () => {
        setFocus(entry.storm);
      },
      listeners,
    );
    entry.row.addEventListener(
      "pointerleave",
      () => {
        setFocus(null);
      },
      listeners,
    );
    button?.addEventListener(
      "focus",
      () => {
        setFocus(entry.storm);
      },
      listeners,
    );
    button?.addEventListener(
      "blur",
      () => {
        setFocus(null);
      },
      listeners,
    );
    button?.addEventListener(
      "click",
      () => {
        // A click on a storm that is not alive at this step turns the watch to its last fix.
        if (positionAt(entry.storm, Math.floor(step)) === null) {
          jump(entry.storm.fixes.at(-1)?.step ?? step, true);
        }
      },
      listeners,
    );
  }

  let big: DrawnChart | null = null;
  function layout(): void {
    const width = Math.max(300, Math.round(bigSvg.parentElement?.clientWidth ?? 900));
    big = drawSeparation(bigSvg, steps, {
      width,
      height: width < 600 ? 300 : 380,
      compact: false,
      lastStep: chartLast,
      initTime,
    });
    big.reveal(chartLast);
    const box = scene.getBoundingClientRect();
    relief?.resize(box.width, box.height);
    placeTags(Math.floor(step));
    redraw();
  }

  show(step);
  setPlaying(false);
  hero.dataset["state"] = "ready";
  await document.fonts.ready;
  const resize = new ResizeObserver(layout);
  resize.observe(scene);
  resize.observe(bigSvg.parentElement ?? bigSvg);

  // The GPU starts after the ledger is on screen and readable.
  await new Promise((resolve) => requestAnimationFrame(resolve));
  const field = manifest.fields.find((candidate) => candidate.name === "mslp");
  try {
    if (field === undefined) throw new Error("the run has no sea-level pressure field");
    const [{ Relief: ReliefClass }, baselineAtlas, forkAtlas] = await Promise.all([
      import("./relief"),
      decodeAtlas(`${base}baseline/mslp.png`),
      decodeAtlas(`${base}fork/mslp.png`),
    ]);
    relief = new ReliefClass({
      canvas,
      atlases: { baseline: baselineAtlas, fork: forkAtlas },
      range: field.range,
      storms: all,
      lastStep,
      inks: [inks.ground, inks.ink],
    });
    const box = scene.getBoundingClientRect();
    relief.resize(box.width, box.height);
    hero.dataset["relief"] = "ready";
  } catch (error) {
    still.hidden = false;
    canvas.hidden = true;
    hero.dataset["relief"] = "still";
    still.title = error instanceof Error ? error.message : "";
    // The still image is the reduced-motion frame, so the ledger holds the same instant.
    run = "fork";
    turned = true;
    show(restStep);
  }
  placeTags(Math.floor(step));

  const seen = new IntersectionObserver(([entry]) => {
    visible = entry?.isIntersecting ?? true;
    if (visible) {
      previous = performance.now();
      redraw();
    }
  });
  seen.observe(scene);
  const onVisibility = (): void => {
    visible = document.visibilityState === "visible";
    if (visible) {
      previous = performance.now();
      redraw();
    }
  };
  document.addEventListener("visibilitychange", onVisibility, listeners);

  if (motion && relief !== null) {
    setPlaying(true);
  }
  previous = performance.now();
  redraw();

  return {
    dispose: () => {
      setPlaying(false);
      if (request !== null) cancelAnimationFrame(request);
      resize.disconnect();
      seen.disconnect();
      events.abort();
      relief?.dispose();
      relief = null;
    },
  };
}
