import { FRAME_COUNT, tileSize, type TileSize } from "./atlas";
import type { Ramp } from "./colours";
import type { FieldMetadata } from "./manifest";

export type FieldKind = "linear" | "sqrt" | "wind";

/** Shader code for each decoding branch. */
export type KindCode = 0 | 1 | 2;

export type ValueRange = readonly [number, number];

export interface FieldSelection {
  readonly kind: KindCode;
  readonly range: ValueRange;
  readonly tile: TileSize;
  readonly diff: 0 | 1;
  readonly ramp: Ramp;
  /** Contour interval in the field's native units; zero draws no contours. */
  readonly contourInterval: number;
}

export interface TextureFiles {
  readonly atlas: string;
  readonly atlasV: string | null;
  readonly base: string | null;
  readonly baseV: string | null;
}

export interface FrameSelection {
  readonly a: number;
  readonly b: number;
  readonly mix: number;
}

export const SPEED_MAX_MPS = 60;

const MSLP_CONTOUR_PA = 4 * 100;
const Z500_CONTOUR_M = 60;
const STANDARD_GRAVITY = 9.80665;

export function fieldKind(field: FieldMetadata): FieldKind {
  return field.files.length === 2 ? "wind" : field.encoding;
}

function kindCode(kind: FieldKind): KindCode {
  if (kind === "wind") {
    return 2;
  }
  return kind === "sqrt" ? 1 : 0;
}

function contourInterval(field: FieldMetadata): number {
  if (field.name === "mslp") {
    return MSLP_CONTOUR_PA;
  }
  if (field.name === "z500") {
    return Z500_CONTOUR_M * STANDARD_GRAVITY;
  }
  return 0;
}

/** Uniform values that follow from the field alone, before a run is chosen. */
export function fieldSelection(field: FieldMetadata, difference: boolean): FieldSelection {
  const kind = fieldKind(field);
  return {
    kind: kindCode(kind),
    range: field.range,
    tile: tileSize(field.resolution_deg),
    diff: difference ? 1 : 0,
    ramp: difference ? "diverging" : kind === "wind" ? "speed" : "scalar",
    contourInterval: contourInterval(field),
  };
}

/** The field's own files, plus the same files from the baseline run in difference mode. */
export function textureFiles(field: FieldMetadata, difference: boolean): TextureFiles {
  const [atlas, atlasV = null] = field.files;
  if (atlas === undefined) {
    throw new Error(`${field.name} has no atlas file`);
  }
  return {
    atlas,
    atlasV,
    base: difference ? atlas : null,
    baseV: difference ? atlasV : null,
  };
}

export function frameSelection(frame: number): FrameSelection {
  const clamped = Math.min(Math.max(frame, 0), FRAME_COUNT - 1);
  const a = Math.floor(clamped);
  const mix = clamped - a;
  return { a, b: mix === 0 ? a : Math.min(a + 1, FRAME_COUNT - 1), mix };
}

function clamp01(value: number): number {
  return Math.min(Math.max(value, 0), 1);
}

export function scalarPosition(value: number, range: ValueRange): number {
  return clamp01((value - range[0]) / (range[1] - range[0]));
}

export function speedPosition(speed: number): number {
  return clamp01(speed / SPEED_MAX_MPS);
}

/**
 * Place zero halfway along the diverging ramp, whatever the field's offsets.
 *
 * The full native span covers each half, so a difference of half the span
 * reaches an end of the ramp and larger differences saturate there.
 */
export function differencePosition(difference: number, range: ValueRange): number {
  return clamp01(0.5 + difference / (range[1] - range[0]));
}
