import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { decode, windSpeed } from "../src/decode";
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
const TP6H: FieldMetadata = {
  name: "tp6h",
  files: ["tp6h"],
  range: [0, 0.1],
  encoding: "sqrt",
  resolution_deg: 1,
};

interface Fixture {
  readonly bytes: ReadonlyArray<ReadonlyArray<number>>;
  readonly decoded: ReadonlyArray<ReadonlyArray<number>>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNumberRows(value: unknown): value is ReadonlyArray<ReadonlyArray<number>> {
  return (
    Array.isArray(value) &&
    value.every((row) => Array.isArray(row) && row.every((item) => typeof item === "number"))
  );
}

function fixture(name: string): Fixture {
  const text = readFileSync(new URL(`../../tests/fixtures/${name}.json`, import.meta.url), "utf8");
  const value: unknown = JSON.parse(text);
  if (!isRecord(value)) {
    throw new Error(`${name} is not an object`);
  }
  const { bytes, decoded } = value;
  if (!isNumberRows(bytes) || !isNumberRows(decoded)) {
    throw new Error(`${name} has no byte rows and decoded rows`);
  }
  return { bytes, decoded };
}

const GOLDEN: ReadonlyArray<readonly [string, FieldMetadata]> = [
  ["sample_mslp", MSLP],
  ["sample_u10", WIND],
  ["sample_v10", WIND],
];

describe("decode", () => {
  it.each(GOLDEN)("matches the exporter's floats for %s within half a step", (name, field) => {
    const data = fixture(name);
    const bytes = data.bytes.flat();
    const reference = data.decoded.flat();
    expect(bytes).toHaveLength(181 * 360);
    expect(reference).toHaveLength(bytes.length);
    const halfStep = (field.range[1] - field.range[0]) / (2 * 255);
    const worst = bytes.reduce(
      (maximum, byte, index) =>
        Math.max(maximum, Math.abs(decode(byte, field) - (reference[index] ?? Number.NaN))),
      0,
    );
    expect(worst).toBeLessThanOrEqual(halfStep + 1e-9);
  });

  it("maps linear endpoints to the field range", () => {
    expect(decode(0, MSLP)).toBeCloseTo(92000, 6);
    expect(decode(255, MSLP)).toBeCloseTo(105000, 6);
    expect(decode(177, MSLP)).toBeCloseTo(101023.5294118, 6);
    expect(decode(0, WIND)).toBeCloseTo(-60, 6);
    expect(decode(255, WIND)).toBeCloseTo(60, 6);
  });

  it("squares the normalised square-root encoding", () => {
    expect(decode(0, TP6H)).toBeCloseTo(0, 12);
    expect(decode(126, TP6H)).toBeCloseTo(0.0244152, 7);
    expect(decode(255, TP6H)).toBeCloseTo(0.1, 12);
  });

  it("gives the wind speed reported for the fixture pair", () => {
    expect(windSpeed(191, 42, WIND)).toBeCloseTo(50.118, 3);
    expect(windSpeed(128, 128, WIND)).toBeCloseTo(0.3327561, 6);
  });
});
