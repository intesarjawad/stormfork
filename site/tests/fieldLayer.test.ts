import { describe, expect, it } from "vitest";

import {
  differencePosition,
  fieldKind,
  fieldSelection,
  frameSelection,
  scalarPosition,
  speedPosition,
  textureFiles,
} from "../src/fieldLayer";
import type { FieldMetadata } from "../src/manifest";

const MSLP: FieldMetadata = {
  name: "mslp",
  files: ["mslp"],
  range: [92000, 105000],
  encoding: "linear",
  resolution_deg: 1,
};
const WIND: FieldMetadata = {
  name: "wind10",
  files: ["u10", "v10"],
  range: [-60, 60],
  encoding: "linear",
  resolution_deg: 1,
};
const WIND_2DEG: FieldMetadata = {
  ...WIND,
  resolution_deg: 2,
};
const TP6H: FieldMetadata = {
  name: "tp6h",
  files: ["tp6h"],
  range: [0, 0.1],
  encoding: "sqrt",
  resolution_deg: 1,
};
const Z500: FieldMetadata = {
  name: "z500",
  files: ["z500"],
  range: [48000, 59000],
  encoding: "linear",
  resolution_deg: 1,
};

describe("field selection", () => {
  it("names the decoding branch of each field kind", () => {
    expect(fieldKind(MSLP)).toBe("linear");
    expect(fieldKind(TP6H)).toBe("sqrt");
    expect(fieldKind(WIND)).toBe("wind");
  });

  it("selects linear decoding, one texture and the pressure contours", () => {
    expect(fieldSelection(MSLP, false)).toEqual({
      kind: 0,
      range: [92000, 105000],
      tile: [361, 181],
      diff: 0,
      ramp: "scalar",
      contourInterval: 400,
    });
    expect(textureFiles(MSLP, false)).toEqual({
      atlas: "mslp",
      atlasV: null,
      base: null,
      baseV: null,
    });
  });

  it("selects square-root decoding for precipitation", () => {
    expect(fieldSelection(TP6H, false).kind).toBe(1);
    expect(fieldSelection(TP6H, false).ramp).toBe("scalar");
    expect(fieldSelection(TP6H, false).contourInterval).toBe(0);
  });

  it("selects both wind textures, the speed ramp and the 2 degree tile", () => {
    expect(fieldSelection(WIND, false)).toEqual({
      kind: 2,
      range: [-60, 60],
      tile: [361, 181],
      diff: 0,
      ramp: "speed",
      contourInterval: 0,
    });
    expect(fieldSelection(WIND_2DEG, false).tile).toEqual([181, 91]);
    expect(textureFiles(WIND, false)).toEqual({
      atlas: "u10",
      atlasV: "v10",
      base: null,
      baseV: null,
    });
  });

  it("draws the 500 hPa height every 60 geopotential metres", () => {
    expect(fieldSelection(Z500, false).contourInterval).toBeCloseTo(588.399, 9);
  });

  it("binds the baseline files only in difference mode", () => {
    expect(fieldSelection(MSLP, true)).toMatchObject({ diff: 1, ramp: "diverging" });
    expect(textureFiles(MSLP, true)).toEqual({
      atlas: "mslp",
      atlasV: null,
      base: "mslp",
      baseV: null,
    });
    expect(textureFiles(WIND, true)).toEqual({
      atlas: "u10",
      atlasV: "v10",
      base: "u10",
      baseV: "v10",
    });
    expect(fieldSelection(WIND, true).ramp).toBe("diverging");
  });
});

describe("field positions", () => {
  it("maps a value into its field range", () => {
    expect(scalarPosition(92000, MSLP.range)).toBe(0);
    expect(scalarPosition(98500, MSLP.range)).toBe(0.5);
    expect(scalarPosition(105000, MSLP.range)).toBe(1);
    expect(scalarPosition(110000, MSLP.range)).toBe(1);
  });

  it("maps wind speed onto the fixed 0 to 60 m/s ramp", () => {
    expect(speedPosition(0)).toBe(0);
    expect(speedPosition(30)).toBe(0.5);
    expect(speedPosition(60)).toBe(1);
    expect(speedPosition(50.118)).toBeCloseTo(0.8353, 4);
  });

  it("puts no change at the centre of the diverging ramp", () => {
    expect(differencePosition(0, MSLP.range)).toBe(0.5);
    expect(differencePosition(6500, MSLP.range)).toBe(1);
    expect(differencePosition(-6500, MSLP.range)).toBe(0);
    expect(differencePosition(0, WIND.range)).toBe(0.5);
  });
});

describe("frame selection", () => {
  it.each([
    [0, { a: 0, b: 0, mix: 0 }],
    [3.5, { a: 3, b: 4, mix: 0.5 }],
    [40, { a: 40, b: 40, mix: 0 }],
    [41, { a: 40, b: 40, mix: 0 }],
    [-2, { a: 0, b: 0, mix: 0 }],
  ])("selects the frames around %s", (frame, expected) => {
    expect(frameSelection(frame)).toEqual(expected);
  });
});
