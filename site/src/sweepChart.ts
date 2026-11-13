import type { SweepStop } from "./manifest";
import type { TrackSample } from "./tracks";

export interface MemberLatitude {
  readonly member: number;
  readonly lat: number;
}

/** One published displacement: the fork's members and the control's spread. */
export interface SweepColumn {
  readonly stop: SweepStop;
  /** Signed kilometres: east and north are positive, west and south negative. */
  readonly offsetKm: number;
  readonly members: readonly MemberLatitude[];
  /** Lowest and highest control latitude, or null when no control is published. */
  readonly control: readonly [number, number] | null;
}

export interface ChartPoint {
  readonly stop: string;
  readonly member: number;
  readonly lat: number;
  readonly x: number;
  readonly y: number;
}

export interface ChartTick {
  readonly label: string;
  readonly position: number;
}

export interface SweepChartLayout {
  readonly width: number;
  readonly height: number;
  /** The drawing area inside the axis margins. */
  readonly plot: {
    readonly left: number;
    readonly right: number;
    readonly top: number;
    readonly bottom: number;
  };
  readonly points: readonly ChartPoint[];
  /** SVG polygon points of the control band; empty with fewer than two stops. */
  readonly band: string;
  readonly xTicks: readonly (ChartTick & { readonly stop: string })[];
  readonly yTicks: readonly ChartTick[];
}

const MARGIN = { left: 44, right: 12, top: 12, bottom: 28 } as const;
const PADDING_FRACTION = 0.05;
const MIN_SPAN_DEG = 1;

/** Run whose members are plotted as points. */
export const FORK_RUN = "fork";
/** Run whose spread is drawn as the band. */
export const CONTROL_RUN = "control";

/**
 * Each member's latitude at one step of a run, in member order.
 *
 * With no step the member's last sample is used. A member without a sample at
 * the step is left out rather than given an invented position.
 */
export function memberLatitudes(
  samples: readonly TrackSample[],
  run: string,
  step: number | null,
): readonly MemberLatitude[] {
  const chosen = new Map<number, TrackSample>();
  for (const sample of samples) {
    if (sample.run !== run) {
      continue;
    }
    if (step !== null) {
      if (sample.step === step) {
        chosen.set(sample.member, sample);
      }
      continue;
    }
    const earlier = chosen.get(sample.member);
    if (earlier === undefined || sample.step > earlier.step) {
      chosen.set(sample.member, sample);
    }
  }
  return [...chosen.values()]
    .sort((a, b) => a.member - b.member)
    .map((sample) => ({ member: sample.member, lat: sample.lat }));
}

export function signedOffsetKm(stop: SweepStop): number {
  return stop.direction === "west" || stop.direction === "south"
    ? -stop.distance_km
    : stop.distance_km;
}

/** Collect one stop's fork latitudes and control range at the landfall step. */
export function sweepColumn(
  stop: SweepStop,
  samples: readonly TrackSample[],
  landfallStep: number | null,
): SweepColumn {
  const control = memberLatitudes(samples, CONTROL_RUN, landfallStep).map((entry) => entry.lat);
  return {
    stop,
    offsetKm: signedOffsetKm(stop),
    members: memberLatitudes(samples, FORK_RUN, landfallStep),
    control: control.length === 0 ? null : [Math.min(...control), Math.max(...control)],
  };
}

function scale(value: number, from: number, to: number, start: number, end: number): number {
  return to === from ? (start + end) / 2 : start + ((value - from) / (to - from)) * (end - start);
}

function latitudeLabel(lat: number): string {
  return lat.toFixed(1);
}

/** Positions for every point, the control band and the axis ticks. */
export function sweepChartLayout(
  columns: readonly SweepColumn[],
  width = 480,
  height = 240,
): SweepChartLayout {
  const plot = {
    left: MARGIN.left,
    right: width - MARGIN.right,
    top: MARGIN.top,
    bottom: height - MARGIN.bottom,
  };
  const ordered = [...columns].sort((a, b) => a.offsetKm - b.offsetKm);
  const values = ordered.flatMap((column) => [
    ...column.members.map((entry) => entry.lat),
    ...(column.control ?? []),
  ]);
  const lowest = values.length === 0 ? 0 : Math.min(...values);
  const highest = values.length === 0 ? 0 : Math.max(...values);
  const pad = Math.max(highest - lowest, MIN_SPAN_DEG) * PADDING_FRACTION;
  const centre = (lowest + highest) / 2;
  const half = Math.max(highest - lowest, MIN_SPAN_DEG) / 2 + pad;
  const south = centre - half;
  const north = centre + half;
  const offsets = ordered.map((column) => column.offsetKm);
  const west = Math.min(...offsets);
  const east = Math.max(...offsets);
  const x = (offsetKm: number): number => scale(offsetKm, west, east, plot.left, plot.right);
  const y = (lat: number): number => scale(lat, south, north, plot.bottom, plot.top);

  const points = ordered.flatMap((column) =>
    column.members.map((entry) => ({
      stop: column.stop.id,
      member: entry.member,
      lat: entry.lat,
      x: x(column.offsetKm),
      y: y(entry.lat),
    })),
  );
  const withControl = ordered.flatMap((column) =>
    column.control === null ? [] : [{ x: x(column.offsetKm), range: column.control }],
  );
  const band =
    withControl.length < 2
      ? ""
      : [
          ...withControl.map((entry) => [entry.x, y(entry.range[1])] as const),
          ...withControl.map((entry) => [entry.x, y(entry.range[0])] as const).reverse(),
        ]
          .map(([px, py]) => `${px.toFixed(2)},${py.toFixed(2)}`)
          .join(" ");
  const yTicks = [lowest, centre, highest]
    .map((lat) => ({ label: latitudeLabel(lat), position: y(lat) }))
    .filter(
      (tick, index, ticks) => ticks.findIndex((other) => other.label === tick.label) === index,
    );
  return {
    width,
    height,
    plot,
    points,
    band,
    xTicks: ordered.map((column) => ({
      stop: column.stop.id,
      label: String(Math.round(column.offsetKm)),
      position: x(column.offsetKm),
    })),
    yTicks: values.length === 0 ? [] : yTicks,
  };
}

function range(values: readonly number[]): string {
  const low = Math.min(...values);
  const high = Math.max(...values);
  return `${latitudeLabel(low)} to ${latitudeLabel(high)} °N`;
}

/** One column in words, for readers who cannot see the chart. */
export function sweepColumnText(column: SweepColumn, label: string): string {
  if (column.members.length === 0) {
    return `${label}: no member has a position at the landfall step.`;
  }
  const members = `${String(column.members.length)} member${column.members.length === 1 ? "" : "s"}`;
  const spread =
    column.control === null
      ? "no control spread is published"
      : `control spread ${range(column.control)}`;
  return `${label}: landfall latitude ${range(column.members.map((entry) => entry.lat))} across ${members}; ${spread}.`;
}
