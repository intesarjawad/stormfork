import { trackColour } from "./colours";
import { dashStyle } from "./dash";
import { svgNode } from "./dom";
import { DEFAULT_FIELD } from "./loadExperiment";
import type { FieldMetadata } from "./manifest";

export const FIELD_LABELS: Readonly<Record<string, string>> = {
  mslp: "Sea-level pressure",
  wind10: "10 m wind speed",
  t2m: "2 m temperature",
  z500: "500 hPa geopotential",
  tp6h: "6 h precipitation",
};

export const RUN_LABELS: Readonly<Record<string, string>> = {
  baseline: "Baseline",
  fork: "Fork",
  control: "Control",
  observed: "Observed",
};

export function fieldLabel(field: FieldMetadata): string {
  return FIELD_LABELS[field.name] ?? field.name;
}

export function renderLegend(legend: HTMLUListElement, runs: readonly string[]): void {
  legend.replaceChildren();
  for (const run of runs) {
    const item = document.createElement("li");
    const swatch = svgNode("svg", { viewBox: "0 0 40 12", "aria-hidden": "true" });
    const style = dashStyle(run);
    swatch.append(
      svgNode("line", {
        x1: "0",
        x2: "40",
        y1: "6",
        y2: "6",
        stroke: trackColour(run),
        "stroke-width": "2.5",
        "stroke-dasharray": style.dash === 0 ? "none" : style.dash > style.gap ? "12 5" : "1.5 5",
        "stroke-linecap": style.dash > 0 && style.dash <= style.gap ? "round" : "butt",
      }),
    );
    if (run === "fork") {
      swatch.append(svgNode("path", { d: "M20 1 L25 6 L20 11 L15 6 Z", fill: trackColour(run) }));
    }
    item.append(swatch, document.createTextNode(RUN_LABELS[run] ?? run));
    legend.append(item);
  }
}

export function webGlAvailable(): boolean {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("webgl2");
  if (context === null) {
    return false;
  }
  context.getExtension("WEBGL_lose_context")?.loseContext();
  return true;
}

export function selectedField(fields: readonly FieldMetadata[], requested: string | null): string {
  return requested !== null && fields.some((field) => field.name === requested)
    ? requested
    : fields.some((field) => field.name === DEFAULT_FIELD)
      ? DEFAULT_FIELD
      : (fields[0]?.name ?? DEFAULT_FIELD);
}

export function selectedRun(runs: readonly string[], requested: string | null): string {
  return requested !== null && runs.includes(requested)
    ? requested
    : runs.includes("baseline")
      ? "baseline"
      : (runs[0] ?? "");
}
