import type { Experiment } from "./loadExperiment";
import type { GlobeRenderer } from "./globeRenderer";
import type { TrackSample } from "./tracks";
import type { Sweep } from "./manifest";
import type { DivergencePoint } from "./divergence";
import { DEFAULT_FIELD } from "./loadExperiment";
import { MAX_STEP } from "./scrubber";

export interface AtlasState {
  experiment: Experiment | null;
  globe: GlobeRenderer | null;
  initTime: string;
  openTicket: number;
  fieldTicket: number;
  stopTicket: number;
  id: string;
  layer: string;
  run: string;
  lead: number | null;
  frame: number;
  difference: boolean;
  record: boolean;
  tracks: readonly TrackSample[];
  trackProblem: unknown;
  fieldProblem: string;
  sweep: Sweep | null;
  divergence: readonly DivergencePoint[];
  stopTracks: Map<number, readonly TrackSample[]>;
  playing: boolean;
  playRequest: number | null;
  playStart: number;
  playFrom: number;
}

export function createAtlasState(): AtlasState {
  return {
    experiment: null,
    globe: null,
    initTime: "",
    openTicket: 0,
    fieldTicket: 0,
    stopTicket: 0,
    id: "",
    layer: DEFAULT_FIELD,
    run: "",
    lead: null,
    frame: 0,
    difference: false,
    record: false,
    tracks: [],
    trackProblem: null,
    fieldProblem: "",
    sweep: null,
    divergence: [],
    stopTracks: new Map(),
    playing: false,
    playRequest: null,
    playStart: 0,
    playFrom: 0,
  };
}

export function maxStep(state: AtlasState): number {
  return Math.min(MAX_STEP, state.experiment?.manifest.steps ?? MAX_STEP);
}
