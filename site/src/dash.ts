import type { Vec3 } from "./tracks";

export interface DashStyle {
  readonly dash: number;
  readonly gap: number;
}

const SOLID: DashStyle = { dash: 0, gap: 0 };
const STYLES: Readonly<Record<string, DashStyle>> = {
  baseline: SOLID,
  fork: { dash: 0.08, gap: 0.04 },
  control: { dash: 0.025, gap: 0.035 },
};

/** Dashing also distinguishes runs, so colour is never the only cue. */
export function dashStyle(run: string): DashStyle {
  return STYLES[run] ?? { dash: 0.05, gap: 0.03 };
}

function clampSegment(value: number): number {
  return Math.min(Math.max(value, 0), 1);
}

function interpolate(a: Vec3, b: Vec3, t: number): Vec3 {
  const ratio = clampSegment(t);
  return [a[0] + (b[0] - a[0]) * ratio, a[1] + (b[1] - a[1]) * ratio, a[2] + (b[2] - a[2]) * ratio];
}

/**
 * Cut a polyline into dash and gap runs measured in world-space length.
 *
 * A zero dash length keeps the polyline solid. Returned pairs are adjacent
 * positions of the same line, so they can be uploaded as line segments.
 */
export function dashSegments(points: readonly Vec3[], dash: number, gap: number): readonly Vec3[] {
  if (points.length < 2) {
    return [];
  }
  if (dash <= 0) {
    return points.slice(0, -1).flatMap((point, index) => {
      const next = points[index + 1];
      return next === undefined ? [] : [point, next];
    });
  }
  const period = dash + gap;
  const segments: Vec3[] = [];
  let travelled = 0;
  for (let index = 0; index < points.length - 1; index += 1) {
    const start = points[index];
    const end = points[index + 1];
    if (start === undefined || end === undefined) {
      continue;
    }
    const length = Math.hypot(end[0] - start[0], end[1] - start[1], end[2] - start[2]);
    if (length === 0) {
      continue;
    }
    let consumed = 0;
    while (consumed < length) {
      const phase = (travelled + consumed) % period;
      const drawing = phase < dash;
      const remaining = drawing ? dash - phase : period - phase;
      const step = Math.min(remaining, length - consumed);
      if (drawing) {
        segments.push(
          interpolate(start, end, consumed / length),
          interpolate(start, end, (consumed + step) / length),
        );
      }
      consumed += step;
    }
    travelled += length;
  }
  return segments;
}
