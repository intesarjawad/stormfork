/** Fixed length of the recorded opening, in milliseconds. */
export const RECORD_STILL_MS = 1_500;
export const RECORD_DRAG_MS = 2_500;
export const RECORD_PLAY_MS = 4_000;
export const RECORD_HOLD_MS = 2_000;
export const RECORD_TOTAL_MS = RECORD_STILL_MS + RECORD_DRAG_MS + RECORD_PLAY_MS + RECORD_HOLD_MS;

export type RecordPhase = "still" | "drag" | "play" | "hold";

export interface RecordFrame {
  readonly phase: RecordPhase;
  /** Fraction of the current phase already played, from 0 to 1. */
  readonly progress: number;
  /** Displacement stop the slider shows. */
  readonly stop: number;
  /** Six-hour step the globe shows. */
  readonly step: number;
  /** Opacity of the "Drag the storm" still, fading to zero. */
  readonly stillOpacity: number;
  readonly done: boolean;
}

function fraction(elapsedMs: number, startMs: number, lengthMs: number): number {
  return Math.min(Math.max((elapsedMs - startMs) / lengthMs, 0), 1);
}

/**
 * State of the opening's fixed ten-second timeline.
 *
 * The still fades over the first 1.5 s, the displacement slider crosses its
 * stops over 2.5 s, playback runs steps 0 to maxStep over 4 s and the last
 * frame holds for 2 s. A missing sweep leaves the drag phase idle.
 */
export function recordFrame(elapsedMs: number, stopCount: number, maxStep: number): RecordFrame {
  const elapsed = Math.max(elapsedMs, 0);
  const lastStop = Math.max(stopCount - 1, 0);
  const dragStart = RECORD_STILL_MS;
  const playStart = dragStart + RECORD_DRAG_MS;
  const holdStart = playStart + RECORD_PLAY_MS;

  if (elapsed < dragStart) {
    const progress = fraction(elapsed, 0, RECORD_STILL_MS);
    return {
      phase: "still",
      progress,
      stop: 0,
      step: 0,
      stillOpacity: 1 - progress,
      done: false,
    };
  }
  if (elapsed < playStart) {
    const progress = fraction(elapsed, dragStart, RECORD_DRAG_MS);
    return {
      phase: "drag",
      progress,
      stop: Math.round(progress * lastStop),
      step: 0,
      stillOpacity: 0,
      done: false,
    };
  }
  if (elapsed < holdStart) {
    const progress = fraction(elapsed, playStart, RECORD_PLAY_MS);
    return {
      phase: "play",
      progress,
      stop: lastStop,
      step: Math.round(progress * maxStep),
      stillOpacity: 0,
      done: false,
    };
  }
  return {
    phase: "hold",
    progress: fraction(elapsed, holdStart, RECORD_HOLD_MS),
    stop: lastStop,
    step: maxStep,
    stillOpacity: 0,
    done: elapsed >= RECORD_TOTAL_MS,
  };
}
