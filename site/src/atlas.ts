/** Atlas layout shared with the exporter: 41 frames on a 5 by 9 tile grid. */
export const FRAME_COUNT = 41;
export const ATLAS_COLUMNS = 5;
export const ATLAS_ROWS = 9;

/** Tile size in texels, including the repeated longitude zero column. */
export type TileSize = readonly [number, number];

/** Normalised atlas coordinates. */
export type AtlasUv = readonly [number, number];

export function tileSize(resolutionDeg: 1 | 2): TileSize {
  return resolutionDeg === 2 ? [181, 91] : [361, 181];
}

export function atlasSize(tile: TileSize): TileSize {
  return [ATLAS_COLUMNS * tile[0], ATLAS_ROWS * tile[1]];
}

function checkFrame(frame: number): void {
  if (!Number.isInteger(frame) || frame < 0 || frame >= FRAME_COUNT) {
    throw new RangeError(`frame must be an integer in [0, ${String(FRAME_COUNT - 1)}]`);
  }
}

/**
 * Texel coordinate of one sample inside a frame's tile.
 *
 * The horizontal axis reaches the repeated zero-longitude column at uv.x = 1,
 * so the last column reproduces the first and the sphere seam is continuous.
 */
export function atlasTexel(frame: number, uv: AtlasUv, tile: TileSize): readonly [number, number] {
  checkFrame(frame);
  const x = (frame % ATLAS_COLUMNS) * tile[0] + 0.5 + uv[0] * (tile[0] - 1);
  const y =
    Math.floor(frame / ATLAS_COLUMNS) * tile[1] +
    0.5 +
    Math.min(Math.max(uv[1] * (tile[1] - 1), 0), tile[1] - 1);
  return [x, y];
}

/** Texel coordinate divided by the atlas size, for a texture lookup. */
export function atlasUv(frame: number, uv: AtlasUv, tile: TileSize): readonly [number, number] {
  const [x, y] = atlasTexel(frame, uv, tile);
  const [width, height] = atlasSize(tile);
  return [x / width, y / height];
}
