import { describe, expect, it } from "vitest";

import { LUT_SIZE, lutIndex, rampColour, rampLut } from "../src/colours";

describe("colour ramps", () => {
  it("maps the diverging ramp from ice blue through the dark centre to red", () => {
    expect(rampColour("diverging", 0)).toEqual([95, 195, 232]);
    expect(rampColour("diverging", 0.5)).toEqual([22, 44, 48]);
    expect(rampColour("diverging", 1)).toEqual([255, 81, 72]);
    expect(rampColour("diverging", 0.25)).toEqual([59, 120, 140]);
  });

  it("interpolates between the scalar ramp stops", () => {
    expect(rampColour("scalar", 0)).toEqual([11, 43, 51]);
    expect(rampColour("scalar", 0.5)).toEqual([40, 128, 132]);
    expect(rampColour("scalar", 1)).toEqual([200, 238, 232]);
    expect(rampColour("scalar", 0.125)).toEqual([17, 65, 73]);
  });

  it("clamps outside the ramp", () => {
    expect(rampColour("speed", -1)).toEqual([11, 43, 51]);
    expect(rampColour("speed", 2)).toEqual([200, 238, 232]);
  });

  it("selects the LUT entry for a normalised value", () => {
    expect(lutIndex(0)).toBe(0);
    expect(lutIndex(0.5)).toBe(128);
    expect(lutIndex(1)).toBe(255);
    expect(lutIndex(-1)).toBe(0);
    expect(lutIndex(2)).toBe(255);
  });

  it("selects entry 213 for the fixture's 50.118 m/s wind", () => {
    expect(lutIndex(50.118 / 60)).toBe(213);
  });

  it("writes opaque RGBA entries with exact endpoints", () => {
    const data = rampLut("diverging");
    expect(data).toHaveLength(LUT_SIZE * 4);
    expect([...data.slice(0, 4)]).toEqual([95, 195, 232, 255]);
    expect([...data.slice(-4)]).toEqual([255, 81, 72, 255]);
    expect(data[3]).toBe(255);
    expect(data[data.length - 1]).toBe(255);
  });

  it("keeps the centre entry near the dark neutral", () => {
    expect([...rampLut("diverging").slice(128 * 4, 128 * 4 + 4)]).toEqual([23, 44, 48, 255]);
  });
});
