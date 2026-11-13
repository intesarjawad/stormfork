import { FRAME_COUNT } from "./atlas";

/** One step is six hours, so the last frame is day 11 at 00:00 UTC. */
export const MAX_STEP = FRAME_COUNT - 1;
export const STEPS_PER_SECOND = 6;
export const EASE_MS = 180;

export type ScrubberAction =
  | { readonly kind: "step"; readonly step: number }
  | { readonly kind: "toggle" }
  | { readonly kind: "ignore" };

function clampStep(step: number, max: number): number {
  return Math.min(Math.max(Math.round(step), 0), max);
}

/** Keep a step inside the published forecast. */
export function clampScrubberStep(step: number, max: number = MAX_STEP): number {
  return clampStep(step, max);
}

/**
 * A timeline label for one six-hour step.
 *
 * Day one starts at 00:00 UTC, so step 10 is day 3 at 12:00 UTC.
 */
export function stepLabel(step: number, max: number = MAX_STEP): string {
  const clamped = clampStep(step, max);
  const day = Math.floor(clamped / 4) + 1;
  const hour = (clamped % 4) * 6;
  return `Day ${String(day)}, ${String(hour).padStart(2, "0")}:00 UTC`;
}

/**
 * The scrubber's keyboard contract.
 *
 * Arrow keys move one step, Page Up and Page Down move four, and Space starts
 * or stops playback. Anything else stays with the browser.
 */
export function scrubberKey(key: string, step: number, max: number = MAX_STEP): ScrubberAction {
  if (key === " " || key === "Space" || key === "Spacebar") {
    return { kind: "toggle" };
  }
  if (key === "ArrowRight" || key === "ArrowUp") {
    return { kind: "step", step: clampStep(step + 1, max) };
  }
  if (key === "ArrowLeft" || key === "ArrowDown") {
    return { kind: "step", step: clampStep(step - 1, max) };
  }
  if (key === "PageUp") {
    return { kind: "step", step: clampStep(step + 4, max) };
  }
  if (key === "PageDown") {
    return { kind: "step", step: clampStep(step - 4, max) };
  }
  return { kind: "ignore" };
}

export interface PlaybackFrame {
  readonly frame: number;
  readonly done: boolean;
}

function ease(t: number): number {
  return t * t * (3 - 2 * t);
}

/**
 * Playback position after `elapsedMs`, moving six steps a second.
 *
 * Fractional frames let the renderer mix neighbouring fields, so a step takes
 * its full 180 ms to settle instead of jumping. Reduced motion skips the mix.
 */
export function playbackFrame(
  elapsedMs: number,
  start: number,
  max: number = MAX_STEP,
  reducedMotion = false,
): PlaybackFrame {
  const begin = clampStep(start, max);
  if (begin >= max) {
    return { frame: max, done: true };
  }
  const raw = begin + (Math.max(elapsedMs, 0) / 1000) * STEPS_PER_SECOND;
  if (raw >= max) {
    return { frame: max, done: true };
  }
  const base = Math.floor(raw);
  const local = raw - base;
  const settled = reducedMotion
    ? local
    : ease(Math.min((local * (1000 / STEPS_PER_SECOND)) / EASE_MS, 1));
  return { frame: Math.min(base + settled, max), done: false };
}
