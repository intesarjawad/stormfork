import type { FieldMetadata } from "./manifest";

const BYTE_MAX = 255;

/**
 * Decode one channel byte exactly as the exporter and the shader do.
 *
 * A mode-L PNG arrives with R = G = B = byte and opaque alpha, so callers pass
 * the red channel; `.r` is therefore byte / 255.
 */
export function decode(byte: number, field: FieldMetadata): number {
  const [lower, upper] = field.range;
  const normalised = byte / BYTE_MAX;
  return field.encoding === "sqrt"
    ? (normalised * Math.sqrt(upper)) ** 2
    : lower + normalised * (upper - lower);
}

/** Wind speed from two encoded components that share one field range. */
export function windSpeed(uByte: number, vByte: number, field: FieldMetadata): number {
  return Math.hypot(decode(uByte, field), decode(vByte, field));
}
