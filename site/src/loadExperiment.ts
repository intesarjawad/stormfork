import { ClampToEdgeWrapping, LinearFilter, NoColorSpace, Texture, TextureLoader } from "three";
import type { FieldMetadata, Manifest } from "./manifest";
import { isManifest } from "./manifest";

/**
 * Every atlas and preview URL is built from this prefix.
 *
 * The exported experiments live beside the site; pointing the prefix at a
 * dataset revision serves the same layout from object storage instead.
 */
export const ATLAS_BASE_URL = "experiments/";

/** The layer drawn before the reader picks another one. */
export const DEFAULT_FIELD = "mslp";

export type RunName = string;

/** Frame-0 previews per run, one texture per file of the field. */
export type Previews = ReadonlyMap<RunName, readonly Texture[]>;

export type PreviewHandler = (previews: Previews) => void;

export type FieldLayer =
  | {
      readonly field: FieldMetadata;
      readonly kind: "ready";
      readonly previews: Previews;
      readonly textures: ReadonlyMap<RunName, readonly Texture[]>;
    }
  | {
      readonly field: FieldMetadata;
      readonly kind: "unavailable";
      /** The first failed request; surfaces the reason without hiding it. */
      readonly error: unknown;
    };

export interface Experiment {
  readonly manifest: Manifest;
  readonly runs: readonly RunName[];
  /** Frame-0 previews of the default layer, ready as soon as loading resolves. */
  readonly previews: Previews;
  /**
   * Load a field's files as a unit. Previews are reported as soon as they are
   * ready so the layer has something to draw while the atlases upload; if any
   * file of the field fails, the layer is reported unavailable instead.
   */
  loadField(fieldName: string, onPreviews?: PreviewHandler): Promise<FieldLayer>;
}

/** Network and texture I/O, injected so the loading rules stay testable. */
export interface LoaderDeps {
  readonly fetchJson: (url: string) => Promise<unknown>;
  readonly loadTexture: (url: string) => Promise<Texture>;
}

function manifestUrl(id: string): string {
  return `${ATLAS_BASE_URL}${id}/manifest.json`;
}

function atlasUrl(id: string, run: RunName, file: string): string {
  return `${ATLAS_BASE_URL}${id}/${run}/${file}.png`;
}

function previewUrl(id: string, run: RunName, file: string): string {
  return `${ATLAS_BASE_URL}${id}/preview/${run}_${file}_f0.png`;
}

async function fetchJson(url: string): Promise<unknown> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} responded ${String(response.status)}`);
  }
  return response.json();
}

async function loadTexture(url: string): Promise<Texture> {
  return new TextureLoader().loadAsync(url);
}

const DEFAULT_DEPS: LoaderDeps = { fetchJson, loadTexture };

async function loadEncodedTexture(url: string, deps: LoaderDeps): Promise<Texture> {
  const texture = await deps.loadTexture(url);
  // The shader reads raw bytes: no transfer function, no premultiplied alpha,
  // and no mip filtering that would bleed across atlas tiles.
  texture.colorSpace = NoColorSpace;
  texture.premultiplyAlpha = false;
  texture.generateMipmaps = false;
  // Atlas rows ascend from south to north; without the flip, texture v = 0
  // would land on the last uploaded row and the globe would be upside down.
  texture.flipY = false;
  texture.minFilter = LinearFilter;
  texture.magFilter = LinearFilter;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
}

async function loadPreviews(
  id: string,
  field: FieldMetadata,
  runs: readonly RunName[],
  deps: LoaderDeps,
): Promise<Previews> {
  const entries = await Promise.all(
    runs.map(async (run) => {
      const textures = await Promise.all(
        field.files.map(async (file) => loadEncodedTexture(previewUrl(id, run, file), deps)),
      );
      return [run, textures] as const;
    }),
  );
  return new Map(entries);
}

async function pickField(
  deps: LoaderDeps,
  id: string,
  field: FieldMetadata,
  runs: readonly RunName[],
  onPreviews?: PreviewHandler,
): Promise<FieldLayer> {
  try {
    const previews = await loadPreviews(id, field, runs, deps);
    onPreviews?.(previews);
    const entries = await Promise.all(
      runs.map(async (run) => {
        // Both components of a field start together and resolve together.
        const textures = await Promise.all(
          field.files.map(async (file) => loadEncodedTexture(atlasUrl(id, run, file), deps)),
        );
        return [run, textures] as const;
      }),
    );
    return { field, kind: "ready", previews, textures: new Map(entries) };
  } catch (error) {
    return { field, kind: "unavailable", error };
  }
}

/**
 * Fetch and validate a manifest, then show its default layer's frame-0 previews.
 *
 * Other layers, atlases included, are requested only when picked.
 *
 * Throws when the manifest request fails, the response is not a manifest, or
 * the default layer's previews cannot be loaded.
 */
export async function loadExperiment(
  id: string,
  deps: LoaderDeps = DEFAULT_DEPS,
): Promise<Experiment> {
  const url = manifestUrl(id);
  const value = await deps.fetchJson(url);
  if (!isManifest(value)) {
    throw new Error(`${url} is not an experiment manifest`);
  }
  const defaultField = value.fields.find((field) => field.name === DEFAULT_FIELD);
  if (defaultField === undefined) {
    throw new Error(`${url} has no ${DEFAULT_FIELD} field`);
  }
  const runs = Object.keys(value.runs);
  const previews = await loadPreviews(id, defaultField, runs, deps);
  return {
    manifest: value,
    runs,
    previews,
    async loadField(fieldName, onPreviews) {
      const field = value.fields.find((candidate) => candidate.name === fieldName);
      if (field === undefined) {
        throw new Error(`${url} has no ${fieldName} field`);
      }
      return pickField(deps, id, field, runs, onPreviews);
    },
  };
}
