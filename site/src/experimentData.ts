import { ATLAS_BASE_URL } from "./loadExperiment";
import { parseDivergence, type DivergencePoint } from "./divergence";
import type { Manifest, SweepStop } from "./manifest";
import { parseReproductions, reproductionText } from "./reproductions";
import { parseTracks, type TrackSample } from "./tracks";

export function leadTimes(manifest: Manifest): readonly number[] {
  if (manifest.kind !== "hindcast" || manifest.steps === undefined) {
    return [];
  }
  return Array.from({ length: manifest.steps + 1 }, (_, index) => index * 6);
}

export function assetUrl(id: string, path: string): string {
  if (/^(?:[a-z]+:)?\/\//i.test(path) || path.startsWith("/")) {
    return path;
  }
  return `${ATLAS_BASE_URL}${id}/${path}`;
}

export async function loadStopTracks(id: string, stop: SweepStop): Promise<readonly TrackSample[]> {
  const url = assetUrl(id, stop.tracks);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} responded ${String(response.status)}`);
  }
  return parseTracks(await response.json());
}

export async function loadStopDivergence(
  id: string,
  path: string,
): Promise<readonly DivergencePoint[]> {
  try {
    const url = assetUrl(id, path);
    const response = await fetch(url);
    if (!response.ok) {
      return [];
    }
    return parseDivergence(await response.json());
  } catch {
    return [];
  }
}

export async function loadReproductionText(id: string): Promise<string | null> {
  try {
    const response = await fetch(assetUrl(id, "reproductions.json"));
    return response.ok ? reproductionText(parseReproductions(await response.json())) : null;
  } catch {
    return null;
  }
}
