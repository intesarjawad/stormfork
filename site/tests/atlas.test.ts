import { describe, expect, it } from "vitest";

import { atlasSize, atlasTexel, atlasUv, FRAME_COUNT, tileSize, type TileSize } from "../src/atlas";

interface Case {
  readonly tile: TileSize;
  readonly frame: number;
  readonly seam: 0 | 1;
  readonly texel: readonly [number, number];
}

const ONE_DEGREE: TileSize = [361, 181];
const TWO_DEGREE: TileSize = [181, 91];

// Expected texels come from the exporter's 5 x 9 tile origins and its repeated
// longitude-zero column, written here rather than recomputed with the mapper.
const CASES: readonly Case[] = [
  { tile: ONE_DEGREE, frame: 0, seam: 0, texel: [0.5, 0.5] },
  { tile: ONE_DEGREE, frame: 0, seam: 1, texel: [360.5, 180.5] },
  { tile: ONE_DEGREE, frame: 4, seam: 0, texel: [1444.5, 0.5] },
  { tile: ONE_DEGREE, frame: 4, seam: 1, texel: [1804.5, 180.5] },
  { tile: ONE_DEGREE, frame: 5, seam: 0, texel: [0.5, 181.5] },
  { tile: ONE_DEGREE, frame: 5, seam: 1, texel: [360.5, 361.5] },
  { tile: ONE_DEGREE, frame: 40, seam: 0, texel: [0.5, 1448.5] },
  { tile: ONE_DEGREE, frame: 40, seam: 1, texel: [360.5, 1628.5] },
  { tile: TWO_DEGREE, frame: 0, seam: 0, texel: [0.5, 0.5] },
  { tile: TWO_DEGREE, frame: 0, seam: 1, texel: [180.5, 90.5] },
  { tile: TWO_DEGREE, frame: 4, seam: 0, texel: [724.5, 0.5] },
  { tile: TWO_DEGREE, frame: 4, seam: 1, texel: [904.5, 90.5] },
  { tile: TWO_DEGREE, frame: 5, seam: 0, texel: [0.5, 91.5] },
  { tile: TWO_DEGREE, frame: 5, seam: 1, texel: [180.5, 181.5] },
  { tile: TWO_DEGREE, frame: 40, seam: 0, texel: [0.5, 728.5] },
  { tile: TWO_DEGREE, frame: 40, seam: 1, texel: [180.5, 818.5] },
];

describe("atlas tiling", () => {
  it("uses the exporter's 41 frames on a five by nine grid", () => {
    expect(FRAME_COUNT).toBe(41);
    expect(atlasSize(ONE_DEGREE)).toEqual([1805, 1629]);
    expect(atlasSize(TWO_DEGREE)).toEqual([905, 819]);
    expect(tileSize(1)).toEqual(ONE_DEGREE);
    expect(tileSize(2)).toEqual(TWO_DEGREE);
  });

  it.each(CASES)("maps frame $frame seam $seam into tile $tile", ({ tile, frame, seam, texel }) => {
    const uv = [seam, seam] as const;
    expect(atlasTexel(frame, uv, tile)).toEqual(texel);
    const normalised = atlasUv(frame, uv, tile);
    const [width, height] = atlasSize(tile);
    expect(normalised[0]).toBeCloseTo(texel[0] / width, 12);
    expect(normalised[1]).toBeCloseTo(texel[1] / height, 12);
  });

  it("clamps the latitude edge to the tile's last row", () => {
    expect(atlasTexel(0, [0, 1.5], ONE_DEGREE)).toEqual([0.5, 180.5]);
    expect(atlasTexel(0, [0, -0.5], ONE_DEGREE)).toEqual([0.5, 0.5]);
  });

  it("rejects a frame outside the atlas", () => {
    expect(() => atlasTexel(-1, [0, 0], ONE_DEGREE)).toThrow("frame");
    expect(() => atlasTexel(41, [0, 0], ONE_DEGREE)).toThrow("frame");
    expect(() => atlasTexel(1.5, [0, 0], ONE_DEGREE)).toThrow("frame");
  });
});
