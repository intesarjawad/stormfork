import type { SweepStop } from "./manifest";

/**
 * Human label for one displacement stop.
 *
 * Distances are the published values; the direction decides the wording so a
 * zero edit reads as the null test instead of "moved 0 km".
 */
export function sweepStopLabel(stop: SweepStop): string {
  if (stop.direction === "none" || stop.distance_km === 0) {
    return "not moved, null test";
  }
  return `Storm moved ${String(Math.round(stop.distance_km))} km ${stop.direction}`;
}
