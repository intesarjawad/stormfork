/** Track separation at one step, in kilometres from the baseline mean. */
export interface DivergencePoint {
  readonly step: number;
  readonly forkKm: number | null;
  readonly controlKm: number | null;
}

export interface SparklinePaths {
  readonly fork: string;
  readonly control: string;
  readonly maxKm: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function separation(value: unknown): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new Error("divergence.json contains an invalid separation");
  }
  return value;
}

/** Read the published per-step separations, ignoring evidence the site does not draw. */
export function parseDivergence(value: unknown): readonly DivergencePoint[] {
  if (!isRecord(value) || !Array.isArray(value["steps"])) {
    throw new Error("divergence.json is not a divergence report");
  }
  return value["steps"].map((item: unknown) => {
    if (!isRecord(item)) {
      throw new Error("divergence.json contains a malformed step");
    }
    const step = item["step"];
    if (typeof step !== "number" || !Number.isInteger(step) || step < 0) {
      throw new Error("divergence.json contains a malformed step");
    }
    return {
      step,
      forkKm: separation(item["fork_sep_km"]),
      controlKm: separation(item["control_sep_km"]),
    };
  });
}

function kilometres(value: number): string {
  return String(Math.round(value));
}

/** Readout that always puts the edit beside the control, its sampling noise. */
export function divergenceText(point: DivergencePoint | null): string {
  if (point === null) {
    return "No divergence is published for this experiment.";
  }
  if (point.forkKm === null && point.controlKm === null) {
    return "No separation is published for this step.";
  }
  if (point.forkKm === null && point.controlKm !== null) {
    return `No fork separation is published; control separation ${kilometres(point.controlKm)} km.`;
  }
  if (point.forkKm !== null && point.controlKm === null) {
    return `Fork separation ${kilometres(point.forkKm)} km; no control separation is published.`;
  }
  if (point.forkKm !== null && point.controlKm !== null) {
    return `Fork separation ${kilometres(point.forkKm)} km; control separation ${kilometres(point.controlKm)} km.`;
  }
  return "No separation is published for this step.";
}

function line(
  points: readonly DivergencePoint[],
  value: (point: DivergencePoint) => number | null,
  width: number,
  height: number,
  maxKm: number,
): string {
  const plotted = points.flatMap((point, index) => {
    const distance = value(point);
    if (distance === null) {
      return [];
    }
    const x = points.length === 1 ? 0 : (index / (points.length - 1)) * width;
    const y = height - (distance / maxKm) * height;
    return [`${x.toFixed(2)},${y.toFixed(2)}`];
  });
  return plotted.join(" ");
}

/** SVG polyline points for both series on one shared scale. */
export function sparklinePaths(
  points: readonly DivergencePoint[],
  width = 240,
  height = 48,
): SparklinePaths {
  const values = points.flatMap((point) =>
    [point.forkKm, point.controlKm].filter((value): value is number => value !== null),
  );
  const maxKm = Math.max(1, ...values);
  return {
    fork: line(points, (point) => point.forkKm, width, height, maxKm),
    control: line(points, (point) => point.controlKm, width, height, maxKm),
    maxKm,
  };
}
