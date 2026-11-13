import { ClampToEdgeWrapping, LinearFilter, NoColorSpace, Texture } from "three";
import { describe, expect, it, vi } from "vitest";

import {
  ATLAS_BASE_URL,
  DEFAULT_FIELD,
  type Experiment,
  type LoaderDeps,
  loadExperiment,
} from "../src/loadExperiment";

const ID = "synthetic";
const RUNS = ["baseline", "fork", "control"];

const EXAMPLE_FIELDS: ReadonlyArray<Record<string, unknown>> = [
  { name: "mslp", files: ["mslp"], range: [92000, 105000], encoding: "linear", resolution_deg: 1 },
  {
    name: "wind10",
    files: ["u10", "v10"],
    range: [-60, 60],
    encoding: "linear",
    resolution_deg: 1,
  },
  { name: "tp6h", files: ["tp6h"], range: [0, 0.1], encoding: "sqrt", resolution_deg: 1 },
];

function exampleManifest(
  fields: ReadonlyArray<Record<string, unknown>> = EXAMPLE_FIELDS,
): Record<string, unknown> {
  const runs = Object.fromEntries(
    RUNS.map((run) => [run, { [`${run}/mslp.png`]: "a".repeat(64) }]),
  );
  return { id: ID, title: "Synthetic comparison", fields, runs };
}

function atlasUrl(run: string, file: string): string {
  return `${ATLAS_BASE_URL}${ID}/${run}/${file}.png`;
}

function previewUrl(run: string, file: string): string {
  return `${ATLAS_BASE_URL}${ID}/preview/${run}_${file}_f0.png`;
}

interface Recording {
  readonly deps: LoaderDeps;
  readonly fetches: string[];
  readonly requests: string[];
}

function recordingDeps(manifest: unknown, fail: (url: string) => boolean = () => false): Recording {
  const fetches: string[] = [];
  const requests: string[] = [];
  return {
    fetches,
    requests,
    deps: {
      fetchJson: (url) => {
        fetches.push(url);
        return Promise.resolve(manifest);
      },
      loadTexture: (url) => {
        requests.push(url);
        return fail(url)
          ? Promise.reject(new Error(`missing ${url}`))
          : Promise.resolve(new Texture());
      },
    },
  };
}

function expectRawBytes(textures: readonly Texture[]): void {
  expect(textures.length).toBeGreaterThan(0);
  for (const texture of textures) {
    expect(texture.colorSpace).toBe(NoColorSpace);
    expect(texture.premultiplyAlpha).toBe(false);
    expect(texture.generateMipmaps).toBe(false);
    expect(texture.flipY).toBe(false);
    expect(texture.minFilter).toBe(LinearFilter);
    expect(texture.magFilter).toBe(LinearFilter);
    expect(texture.wrapS).toBe(ClampToEdgeWrapping);
    expect(texture.wrapT).toBe(ClampToEdgeWrapping);
    expect(texture.version).toBe(1);
  }
}

function readyTextures(layer: Awaited<ReturnType<Experiment["loadField"]>>) {
  if (layer.kind !== "ready") {
    throw new Error(`expected a ready layer, got ${String(layer.error)}`);
  }
  return RUNS.flatMap((run) => layer.textures.get(run) ?? []);
}

describe("loadExperiment", () => {
  it("builds every URL from the exported prefix", () => {
    expect(ATLAS_BASE_URL).toBe("experiments/");
    expect(DEFAULT_FIELD).toBe("mslp");
  });

  it("shows the default layer's previews before any atlas", async () => {
    const { deps, fetches, requests } = recordingDeps(exampleManifest());
    const experiment = await loadExperiment(ID, deps);
    expect(fetches).toEqual([`${ATLAS_BASE_URL}${ID}/manifest.json`]);
    expect(requests).toEqual(RUNS.map((run) => previewUrl(run, "mslp")));
    expect(experiment.runs).toEqual(RUNS);
    expect(experiment.manifest.id).toBe(ID);
    expect([...experiment.previews.keys()]).toEqual(RUNS);
    expectRawBytes([...experiment.previews.values()].flat());
  });

  it("rejects a response that is not a manifest before loading textures", async () => {
    const { deps, requests } = recordingDeps({ id: ID });
    await expect(loadExperiment(ID, deps)).rejects.toThrow("is not an experiment manifest");
    expect(requests).toEqual([]);
  });

  it("rejects a manifest without the default layer", async () => {
    const fields = EXAMPLE_FIELDS.filter((field) => field["name"] !== DEFAULT_FIELD);
    const { deps } = recordingDeps(exampleManifest(fields));
    await expect(loadExperiment(ID, deps)).rejects.toThrow("has no mslp field");
  });

  it("loads both wind files in parallel and resolves only once both are uploaded", async () => {
    const gates = new Map<string, PromiseWithResolvers<Texture>>();
    const requests: string[] = [];
    const deps: LoaderDeps = {
      fetchJson: () => Promise.resolve(exampleManifest()),
      loadTexture: (url) => {
        requests.push(url);
        // Previews resolve at once; atlases stay pending until the test opens them.
        if (url.includes("/preview/")) {
          return Promise.resolve(new Texture());
        }
        const gate = Promise.withResolvers<Texture>();
        gates.set(url, gate);
        return gate.promise;
      },
    };
    const experiment = await loadExperiment(ID, deps);
    const initial = requests.length;

    let settled = false;
    const previewsSeen: Texture[][] = [];
    const pick = experiment
      .loadField("wind10", (previews) => {
        previewsSeen.push([...previews.values()].flat());
      })
      .then((layer) => {
        settled = true;
        return layer;
      });

    await vi.waitFor(() => {
      expect(requests.filter((url) => url.endsWith("/v10.png"))).toHaveLength(RUNS.length);
    });
    const picked = requests.slice(initial);
    for (const run of RUNS) {
      expect(picked.indexOf(atlasUrl(run, "u10"))).toBeLessThan(
        picked.indexOf(atlasUrl(run, "v10")),
      );
      expect(picked).toContain(atlasUrl(run, "v10"));
    }
    expect(settled).toBe(false);
    expect(previewsSeen).toHaveLength(1);
    expect(previewsSeen[0]).toHaveLength(RUNS.length * 2);

    for (const gate of gates.values()) {
      gate.resolve(new Texture());
    }
    const layer = await pick;
    expect(layer.kind).toBe("ready");
    expectRawBytes(readyTextures(layer));
  });

  it("reports the layer unavailable when either file fails", async () => {
    const { deps, requests } = recordingDeps(exampleManifest(), (url) =>
      url.endsWith("/baseline/v10.png"),
    );
    const experiment = await loadExperiment(ID, deps);
    const layer = await experiment.loadField("wind10");
    expect(layer.kind).toBe("unavailable");
    expect(requests).toContain(atlasUrl("baseline", "u10"));
    expect(requests).toContain(atlasUrl("fork", "v10"));
  });

  it("rejects a pick of a field the manifest does not publish", async () => {
    const { deps } = recordingDeps(exampleManifest());
    const experiment = await loadExperiment(ID, deps);
    await expect(experiment.loadField("rh2m")).rejects.toThrow("has no rh2m field");
  });
});
