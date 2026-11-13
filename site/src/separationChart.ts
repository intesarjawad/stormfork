import { svgNode as node } from "./dom";
import { stepTime, verdict, type StepStat } from "./steps";

export interface ChartOptions {
  readonly width: number;
  readonly height: number;
  /** A compact chart is the plate's rail: no axes, only the series and the test row. */
  readonly compact: boolean;
  readonly lastStep: number;
  readonly initTime: string;
}

export interface DrawnChart {
  /** Show the chart as far as `step`; later steps stay in the markup but dim. */
  readonly reveal: (step: number) => void;
  readonly x: (step: number) => number;
}

const KM_TICKS = [0, 100, 500, 1000, 2000, 3000];
const COMPACT_TICKS = [0, 1000, 3000];
let uniqueId = 0;

/**
 * Fork and control separation from the baseline by step, on a square-root scale so
 * the control's few tens of kilometres stay visible beside the fork's thousands, with
 * one cell per step for the outcome of the paired test.
 */
export function drawSeparation(
  svg: SVGSVGElement,
  steps: readonly StepStat[],
  options: ChartOptions,
): DrawnChart {
  const { width, height, compact, lastStep } = options;
  const shown = steps.filter((stat) => stat.step <= lastStep);
  const left = compact ? (width < 520 ? 58 : 100) : 100;
  const right = compact ? 26 : 18;
  const top = compact ? 8 : 20;
  const rowHeight = compact ? 5 : 12;
  const bottom = compact ? rowHeight + 22 : rowHeight + 44;
  const plotHeight = height - top - bottom;
  const peak = Math.max(1, ...shown.map((stat) => stat.forkKm ?? 0));
  const x = (step: number): number =>
    left + (step / Math.max(1, lastStep)) * (width - left - right);
  const y = (km: number): number => top + plotHeight * (1 - Math.sqrt(Math.min(km, peak) / peak));
  const clipId = `reveal-${String(uniqueId)}`;
  uniqueId += 1;

  svg.setAttribute("viewBox", `0 0 ${String(width)} ${String(height)}`);
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));

  const points = (pick: (stat: StepStat) => number | null): string =>
    shown
      .flatMap((stat) => {
        const value = pick(stat);
        return value === null ? [] : [`${x(stat.step).toFixed(1)},${y(value).toFixed(1)}`];
      })
      .join(" ");

  const children: SVGElement[] = [];
  const clip = node("clipPath", { id: clipId });
  const reveal = node("rect", { x: "0", y: "0", width: "0", height: String(height) });
  clip.append(reveal);
  children.push(clip);

  const axes = node("g", { class: "sc-axes" });
  for (const km of (compact ? COMPACT_TICKS : KM_TICKS).filter((tick) => tick <= peak)) {
    {
      axes.append(
        node("line", {
          x1: String(left),
          x2: String(width - right),
          y1: String(y(km)),
          y2: String(y(km)),
          class: "sc-rule",
        }),
        node(
          "text",
          { x: String(left - 10), y: String(y(km) + 4), "text-anchor": "end" },
          km.toLocaleString("en-GB"),
        ),
      );
    }
  }
  if (!compact) {
    for (let step = 0; step <= lastStep; step += 4) {
      const { day, hour } = stepTime(options.initTime, step);
      axes.append(
        node("line", {
          x1: String(x(step)),
          x2: String(x(step)),
          y1: String(top),
          y2: String(top + plotHeight),
          class: "sc-day",
        }),
        node(
          "text",
          {
            x: String(x(step)),
            y: String(top + plotHeight + rowHeight + 30),
            "text-anchor": "middle",
          },
          step === 0 ? `${day}, ${hour}` : day,
        ),
      );
    }
    axes.append(
      node(
        "text",
        { x: String(left), y: "10", class: "sc-unit" },
        "km from baseline, square-root scale",
      ),
    );
  }
  children.push(axes);

  const baselineY = y(0);
  const series = node("g", { "clip-path": `url(#${clipId})` });
  series.append(
    node("line", {
      x1: String(left),
      x2: String(width - right),
      y1: String(baselineY),
      y2: String(baselineY),
      class: "sc-baseline",
    }),
    node("polyline", { points: points((stat) => stat.controlKm), class: "sc-control" }),
    node("polyline", { points: points((stat) => stat.forkKm), class: "sc-fork" }),
  );
  children.push(series);

  const cells = node("g", { class: "sc-cells" });
  const cellY = top + plotHeight + (compact ? 8 : 14);
  const cellWidth = Math.max(3, ((width - left - right) / Math.max(1, lastStep)) * 0.5);
  const cellNodes = new Map<number, SVGElement>();
  for (const stat of shown) {
    const outcome = verdict(stat);
    const cx = x(stat.step);
    const cell =
      outcome === "too-few-pairs"
        ? node("path", {
            d: `M${String(cx - 3)} ${String(cellY)} l6 ${String(rowHeight)} m0 ${String(-rowHeight)} l-6 ${String(rowHeight)}`,
            class: `sc-cell sc-${outcome}`,
          })
        : node("rect", {
            x: String(cx - cellWidth / 2),
            y: String(cellY),
            width: String(cellWidth),
            height: String(rowHeight),
            class: `sc-cell sc-${outcome}`,
          });
    cellNodes.set(stat.step, cell);
    cells.append(cell);
  }
  children.push(cells);
  children.push(
    node(
      "text",
      {
        x: String(left - 10),
        y: String(cellY + rowHeight),
        "text-anchor": "end",
        class: "sc-unit",
      },
      width < 520 ? "test" : "paired test",
    ),
  );

  if (compact) {
    const days = node("g", { class: "sc-axes" });
    for (let step = 0; step <= lastStep; step += width < 520 ? 8 : 4) {
      days.append(
        node(
          "text",
          { x: String(x(step)), y: String(cellY + rowHeight + 14), "text-anchor": "middle" },
          stepTime(options.initTime, step).day,
        ),
      );
    }
    days.append(
      node(
        "text",
        { x: String(left + 12), y: "11", class: "sc-unit" },
        width < 520 ? "km from baseline" : "km from baseline, square-root scale",
      ),
    );
    children.push(days);
  }

  const cursor = node("line", {
    y1: String(top),
    y2: String(cellY + rowHeight + 2),
    class: "sc-cursor",
  });
  const markFork = node("path", { class: "sc-mark" });
  children.push(cursor, markFork);

  svg.replaceChildren(...children);

  const forkAt = new Map(shown.map((stat) => [stat.step, stat.forkKm] as const));
  return {
    x,
    reveal: (step) => {
      const at = Math.min(Math.max(step, 0), lastStep);
      const whole = Math.floor(at);
      reveal.setAttribute("width", String(x(at) + 1));
      cursor.setAttribute("x1", String(x(at)));
      cursor.setAttribute("x2", String(x(at)));
      const km = forkAt.get(whole) ?? null;
      if (km === null) {
        markFork.setAttribute("d", "");
      } else {
        const cx = x(whole);
        const cy = y(km);
        markFork.setAttribute("d", `M${String(cx)} ${String(cy - 5)} l5 5 l-5 5 l-5 -5 z`);
      }
      for (const [index, cell] of cellNodes) {
        cell.classList.toggle("is-future", index > whole);
      }
    },
  };
}
