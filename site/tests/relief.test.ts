import { describe, expect, it } from "vitest";

import { heatAt } from "../src/relief";

describe("heatAt", () => {
  it("cools through the composite's bands in order and never rises", () => {
    const band = (heat: number): string =>
      heat > 0.92 ? "head" : heat > 0.66 ? "inverted" : heat > 0.3 ? "embers" : "char";
    expect([0, 0.5, 3, 10].map((age) => band(heatAt(age)))).toEqual([
      "head",
      "inverted",
      "embers",
      "char",
    ]);
    for (let age = 0; age < 30; age += 0.25) {
      expect(heatAt(age + 0.25)).toBeLessThanOrEqual(heatAt(age));
    }
    expect(heatAt(100)).toBeGreaterThan(0.06);
  });
});
