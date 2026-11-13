import { existsSync, readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { fieldSelection, textureFiles } from "../src/fieldLayer";
import { isManifest } from "../src/manifest";
import { parseTracks } from "../src/tracks";

const DIRECTORY = new URL("../public/experiments/demo-fixture/", import.meta.url);

function read(name: string): unknown {
  return JSON.parse(readFileSync(new URL(name, DIRECTORY), "utf8"));
}

describe("demo fixture", () => {
  it("passes the site's manifest guard", () => {
    const value = read("manifest.json");
    expect(isManifest(value)).toBe(true);
    if (!isManifest(value)) {
      return;
    }
    expect(value.id).toBe("demo-fixture");
    expect(value.runs["baseline"]).toBeDefined();
    expect(value.runs["fork"]).toBeDefined();

    const wind = value.fields.find((field) => field.name === "wind10");
    expect(wind).toBeDefined();
    if (wind !== undefined) {
      expect(fieldSelection(wind, false).tile).toEqual([181, 91]);
      expect(textureFiles(wind, false)).toMatchObject({ atlas: "u10", atlasV: "v10" });
    }
  });

  it("publishes the fallback frame and both runs' tracks", () => {
    expect(existsSync(new URL("fallback/frame20_mslp.png", DIRECTORY))).toBe(true);
    const tracks = parseTracks(read("tracks.json"));
    expect(tracks).toHaveLength(18);
    expect([...new Set(tracks.map((sample) => sample.run))]).toEqual(["baseline", "fork"]);
  });
});
