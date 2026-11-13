import { describe, expect, it } from "vitest";
import { selectedField, selectedRun } from "../src/atlasPresentation";
import type { FieldMetadata } from "../src/manifest";

const pressure: FieldMetadata = {
  name: "mslp",
  files: ["mslp"],
  range: [92000, 105000],
  encoding: "linear",
  resolution_deg: 1,
};
const wind: FieldMetadata = {
  name: "wind10",
  files: ["u10", "v10"],
  range: [-60, 60],
  encoding: "linear",
  resolution_deg: 2,
};

describe("atlas selection", () => {
  it("keeps a published field and falls back to pressure before the first field", () => {
    expect(selectedField([wind, pressure], "wind10")).toBe("wind10");
    expect(selectedField([wind, pressure], "unpublished")).toBe("mslp");
    expect(selectedField([wind, pressure], null)).toBe("mslp");
    expect(selectedField([wind], null)).toBe("wind10");
    expect(selectedField([], null)).toBe("mslp");
  });

  it("keeps a published run and falls back to baseline before the first run", () => {
    expect(selectedRun(["fork", "baseline"], "fork")).toBe("fork");
    expect(selectedRun(["fork", "baseline"], "unknown")).toBe("baseline");
    expect(selectedRun(["fork", "baseline"], null)).toBe("baseline");
    expect(selectedRun(["fork"], null)).toBe("fork");
    expect(selectedRun([], null)).toBe("");
  });
});
