/** Decoding contract for one exported field, as written by the quantiser. */
export type Encoding = "linear" | "sqrt";

export interface FieldMetadata {
  /** Field identity, such as `wind10`. */
  readonly name: string;
  /** Atlas basenames, in shader channel order; wind lists `u10` then `v10`. */
  readonly files: readonly string[];
  /** Fixed native-unit range shared by every file of the field. */
  readonly range: readonly [number, number];
  readonly encoding: Encoding;
  readonly resolution_deg: 1 | 2;
}

export type SweepDirection = "east" | "west" | "north" | "south" | "none";

/**
 * One storm displacement in a published sweep.
 *
 * `tracks` and the optional atlas and divergence paths are relative to the
 * experiment directory, so a sweep can be moved with its manifest. `sha256`
 * hashes the files only this stop publishes; the null stop has none.
 */
export interface SweepStop {
  readonly id: string;
  readonly direction: SweepDirection;
  readonly distance_km: number;
  readonly tracks: string;
  readonly atlas?: string;
  readonly divergence?: string;
  readonly sha256?: Readonly<Record<string, string>>;
}

/**
 * An ordered set of starting-position edits for the hero globe.
 *
 * `landfall_step` is the step whose member latitudes the displacement chart
 * plots; without it each member's last sample is used. `baseline` is the id
 * of the null stop.
 */
export interface Sweep {
  readonly name: string;
  readonly baseline: string;
  readonly landfall_step?: number;
  readonly stops: readonly SweepStop[];
}

/** The only manifest version this site reads; the exporter writes the same. */
export const SCHEMA_VERSION = 2;

/**
 * The target storm at the init. Before the storm's first IBTrACS record it
 * has no position, so `lat` and `lon` are both null there and nowhere else.
 */
export interface StormAtInit {
  readonly ibtracs_sid: string;
  readonly lat: number | null;
  readonly lon: number | null;
  /** ISO time of the storm's first six-hourly record. */
  readonly first_record_time: string;
}

export interface InitialState {
  readonly time: string;
  readonly storm: StormAtInit;
}

/**
 * The manifest keys the site reads.
 *
 * Provenance, tracks and hashes are validated when the experiment is exported;
 * the site reads only what it must to build URLs and decode bytes.
 */
export interface Manifest {
  readonly schema_version?: typeof SCHEMA_VERSION;
  readonly id: string;
  readonly title: string;
  readonly init?: InitialState;
  readonly kind?: "fork" | "sweep" | "hindcast";
  /** Number of six-hour steps in a hindcast, when published. */
  readonly steps?: number;
  readonly fields: readonly FieldMetadata[];
  /** Run name to the relative asset paths published for that run. */
  readonly runs: Readonly<Record<string, Readonly<Record<string, string>>>>;
  /** Present when the hero globe offers a displacement slider; the exporter writes null otherwise. */
  readonly sweep?: Sweep | null;
  /** Provenance the fork page prints beside the plate; validated by the exporter, not here. */
  readonly model?: { readonly name: string; readonly checkpoint: string; readonly md5: string };
  readonly runtime?: { readonly kind: string; readonly device: string | null };
  readonly perturbation?: {
    readonly type: string;
    readonly params: Readonly<Record<string, number>>;
  };
  readonly seeds?: { readonly baseline: number; readonly control: number };
  readonly members?: number;
  readonly limits?: readonly string[];
  readonly created?: string;
}

const WIND_FIELD = "wind10";
const WIND_FILES: readonly string[] = ["u10", "v10"];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isText(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isIndex(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0;
}

const SWEEP_DIRECTIONS: readonly SweepDirection[] = ["east", "west", "north", "south", "none"];

function isSweepDirection(value: unknown): value is SweepDirection {
  return SWEEP_DIRECTIONS.includes(value as SweepDirection);
}

function isSweepStop(value: unknown): value is SweepStop {
  if (!isRecord(value)) {
    return false;
  }
  const { id, direction, distance_km: distance, tracks, atlas, divergence, sha256 } = value;
  if (!isText(id) || !isSweepDirection(direction) || !isFiniteNumber(distance) || distance < 0) {
    return false;
  }
  if (!isText(tracks)) {
    return false;
  }
  if (sha256 !== undefined && !(isRecord(sha256) && Object.values(sha256).every(isText))) {
    return false;
  }
  return (atlas === undefined || isText(atlas)) && (divergence === undefined || isText(divergence));
}

function isSweep(value: unknown): value is Sweep {
  if (!isRecord(value)) {
    return false;
  }
  const { name, baseline, stops, landfall_step: landfallStep } = value;
  if (!isText(name) || !isText(baseline) || !Array.isArray(stops) || stops.length === 0) {
    return false;
  }
  if (landfallStep !== undefined && !isIndex(landfallStep)) {
    return false;
  }
  if (!stops.every(isSweepStop)) {
    return false;
  }
  const ids = new Set(stops.map((stop) => stop.id));
  return ids.size === stops.length && ids.has(baseline);
}

function isStormAtInit(value: unknown): value is StormAtInit {
  if (!isRecord(value)) {
    return false;
  }
  const { ibtracs_sid: sid, lat, lon, first_record_time: firstRecord } = value;
  if (!isText(sid) || !isText(firstRecord)) {
    return false;
  }
  if (lat === null || lon === null) {
    return lat === null && lon === null;
  }
  return isFiniteNumber(lat) && isFiniteNumber(lon) && Math.abs(lat) <= 90;
}

function isInitialState(value: unknown): value is InitialState {
  if (!isRecord(value)) {
    return false;
  }
  const { time, storm } = value;
  return isText(time) && isStormAtInit(storm);
}

function isFieldMetadata(value: unknown): value is FieldMetadata {
  if (!isRecord(value)) {
    return false;
  }
  const { name, files, range, encoding, resolution_deg: resolution } = value;
  if (!isText(name) || !Array.isArray(files) || !files.every(isText)) {
    return false;
  }
  if (name === WIND_FIELD) {
    // A wind layer is only drawable when both components are present.
    if (files.length !== WIND_FILES.length || files[0] !== "u10" || files[1] !== "v10") {
      return false;
    }
  } else if (files.length !== 1) {
    return false;
  }
  if (!Array.isArray(range) || range.length !== 2) {
    return false;
  }
  const lower: unknown = range[0];
  const upper: unknown = range[1];
  if (!isFiniteNumber(lower) || !isFiniteNumber(upper) || lower >= upper) {
    return false;
  }
  if (encoding !== "linear" && encoding !== "sqrt") {
    return false;
  }
  return resolution === 1 || resolution === 2;
}

/**
 * Accept a manifest produced by the exporter and reject anything the site
 * cannot decode, before a single atlas request is made.
 */
export function isManifest(value: unknown): value is Manifest {
  if (!isRecord(value)) {
    return false;
  }
  const { schema_version: version, id, title, fields, runs, kind, steps, sweep, init } = value;
  if (!isText(id) || !isText(title)) {
    return false;
  }
  if (version !== undefined && version !== SCHEMA_VERSION) {
    return false;
  }
  if (init !== undefined && !isInitialState(init)) {
    return false;
  }
  if (kind !== undefined && kind !== "fork" && kind !== "sweep" && kind !== "hindcast") {
    return false;
  }
  if (steps !== undefined && !isIndex(steps)) {
    return false;
  }
  if (sweep !== undefined && sweep !== null && !isSweep(sweep)) {
    return false;
  }
  if (!Array.isArray(fields) || fields.length === 0 || !fields.every(isFieldMetadata)) {
    return false;
  }
  if (!isRecord(runs) || Object.keys(runs).length === 0) {
    return false;
  }
  return Object.entries(runs).every(([run, assets]) => {
    return (
      isText(run) &&
      isRecord(assets) &&
      Object.keys(assets).length > 0 &&
      Object.values(assets).every(isText)
    );
  });
}
