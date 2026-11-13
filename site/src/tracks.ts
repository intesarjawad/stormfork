import { ATLAS_BASE_URL } from "./loadExperiment";

export type Vec3 = readonly [number, number, number];

export interface TrackPoint {
  readonly lat: number;
  readonly lon: number;
}

export interface TrackSample extends TrackPoint {
  readonly run: string;
  readonly member: number;
  readonly step: number;
}

export const SURFACE_RADIUS = 1;

/** Run name under which a hindcast's IBTrACS truth sits beside the forecast runs. */
export const OBSERVED_RUN = "observed";

/** Longitude in [0, 360), matching the atlas and the sphere's texture seam. */
export function wrapLongitude(lon: number): number {
  return ((lon % 360) + 360) % 360;
}

/**
 * Place a geographic point on the sphere used by `SphereGeometry`.
 *
 * Longitude zero sits at -x and longitude 90 at +z, so `uv.x = lon / 360`
 * lands on the same meridian as the atlas.
 */
export function lonLatToPosition(lat: number, lon: number, radius = SURFACE_RADIUS): Vec3 {
  const latitude = (lat * Math.PI) / 180;
  const longitude = (wrapLongitude(lon) * Math.PI) / 180;
  const ring = radius * Math.cos(latitude);
  return [-ring * Math.cos(longitude), radius * Math.sin(latitude), ring * Math.sin(longitude)];
}

/** Flat xyz positions for a polyline, ready for a BufferGeometry. */
export function trackPositions(points: readonly TrackPoint[], radius = SURFACE_RADIUS): number[] {
  return points.flatMap((point) => [...lonLatToPosition(point.lat, point.lon, radius)]);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isText(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function isIndex(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0;
}

function isTrackSample(value: unknown): value is TrackSample {
  if (!isRecord(value)) {
    return false;
  }
  const { run, member, step, lat, lon } = value;
  return (
    isText(run) &&
    isIndex(member) &&
    isIndex(step) &&
    typeof lat === "number" &&
    Number.isFinite(lat) &&
    lat >= -90 &&
    lat <= 90 &&
    typeof lon === "number" &&
    Number.isFinite(lon)
  );
}

/** Read the canonical track records published beside an experiment. */
export function parseTracks(value: unknown): readonly TrackSample[] {
  if (!Array.isArray(value)) {
    throw new Error("tracks.json is not an array");
  }
  return value.map((item) => {
    if (!isTrackSample(item)) {
      throw new Error("tracks.json contains a malformed sample");
    }
    return item;
  });
}

/** The observed track in step order; empty when the experiment is not a hindcast. */
export function observedTrack(samples: readonly TrackSample[]): readonly TrackSample[] {
  return samples.filter((sample) => sample.run === OBSERVED_RUN).sort((a, b) => a.step - b.step);
}

/** The samples of the forecast runs, without the observed truth. */
export function forecastSamples(samples: readonly TrackSample[]): readonly TrackSample[] {
  return samples.filter((sample) => sample.run !== OBSERVED_RUN);
}

export async function loadTracks(id: string): Promise<readonly TrackSample[]> {
  const url = `${ATLAS_BASE_URL}${id}/tracks.json`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} responded ${String(response.status)}`);
  }
  return parseTracks(await response.json());
}
