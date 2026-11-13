import { describe, expect, it } from "vitest";

import { dashSegments, dashStyle } from "../src/dash";
import type { Vec3 } from "../src/tracks";

const LINE: readonly Vec3[] = [
  [0, 0, 0],
  [10, 0, 0],
];

const BENT: readonly Vec3[] = [
  [0, 0, 0],
  [5, 0, 0],
  [5, 5, 0],
];

describe("dash patterns", () => {
  it("cuts a straight line into three-unit dashes and two-unit gaps", () => {
    expect(dashSegments(LINE, 3, 2)).toEqual([
      [0, 0, 0],
      [3, 0, 0],
      [5, 0, 0],
      [8, 0, 0],
    ]);
  });

  it("keeps the final partial dash", () => {
    expect(
      dashSegments(
        [
          [0, 0, 0],
          [12, 0, 0],
        ],
        3,
        2,
      ),
    ).toEqual([
      [0, 0, 0],
      [3, 0, 0],
      [5, 0, 0],
      [8, 0, 0],
      [10, 0, 0],
      [12, 0, 0],
    ]);
  });

  it("carries the pattern around a bend", () => {
    expect(dashSegments(BENT, 4, 1)).toEqual([
      [0, 0, 0],
      [4, 0, 0],
      [5, 0, 0],
      [5, 4, 0],
    ]);
  });

  it("keeps a zero dash length solid", () => {
    expect(dashSegments(BENT, 0, 0)).toEqual([
      [0, 0, 0],
      [5, 0, 0],
      [5, 0, 0],
      [5, 5, 0],
    ]);
    expect(dashSegments([[0, 0, 0]], 3, 2)).toEqual([]);
  });

  it("distinguishes runs without relying on colour alone", () => {
    expect(dashStyle("baseline").dash).toBe(0);
    expect(dashStyle("fork").dash).toBeGreaterThan(0);
    expect(dashStyle("control").dash).toBeGreaterThan(0);
    expect(dashStyle("control").gap).not.toBe(dashStyle("fork").gap);
    expect(dashStyle("unknown").dash).toBeGreaterThan(0);
  });
});
