import type { GlobeRenderer } from "./globeRenderer";

/**
 * Read one fragment of the field material at a geographic point.
 *
 * Kept as a function over the renderer so browser tests use the same material
 * the globe draws.
 */
export function probeField(
  layer: GlobeRenderer,
  frame: number,
  lat: number,
  lon: number,
): readonly [number, number, number, number] {
  return layer.probeField(frame, lat, lon);
}
