import { describe, expect, it } from "vitest";

import { MAX_STEP, playbackFrame, scrubberKey, stepLabel } from "../src/scrubber";

describe("stepLabel", () => {
  it("names each six-hour step in UTC", () => {
    expect(stepLabel(0)).toBe("Day 1, 00:00 UTC");
    expect(stepLabel(1)).toBe("Day 1, 06:00 UTC");
    expect(stepLabel(4)).toBe("Day 2, 00:00 UTC");
    expect(stepLabel(10)).toBe("Day 3, 12:00 UTC");
    expect(stepLabel(MAX_STEP)).toBe("Day 11, 00:00 UTC");
  });

  it("clamps out-of-range steps", () => {
    expect(stepLabel(-1)).toBe("Day 1, 00:00 UTC");
    expect(stepLabel(MAX_STEP + 1)).toBe("Day 11, 00:00 UTC");
  });
});

describe("scrubberKey", () => {
  it("moves one step with the arrow keys", () => {
    expect(scrubberKey("ArrowRight", 3)).toEqual({ kind: "step", step: 4 });
    expect(scrubberKey("ArrowUp", 3)).toEqual({ kind: "step", step: 4 });
    expect(scrubberKey("ArrowLeft", 3)).toEqual({ kind: "step", step: 2 });
    expect(scrubberKey("ArrowDown", 3)).toEqual({ kind: "step", step: 2 });
  });

  it("moves four steps with Page Up and Page Down", () => {
    expect(scrubberKey("PageUp", 3)).toEqual({ kind: "step", step: 7 });
    expect(scrubberKey("PageDown", 7)).toEqual({ kind: "step", step: 3 });
  });

  it("toggles playback with Space", () => {
    expect(scrubberKey(" ", 3)).toEqual({ kind: "toggle" });
    expect(scrubberKey("Space", 3)).toEqual({ kind: "toggle" });
  });

  it("clamps at both ends and leaves other keys alone", () => {
    expect(scrubberKey("ArrowLeft", 0)).toEqual({ kind: "step", step: 0 });
    expect(scrubberKey("PageUp", MAX_STEP)).toEqual({ kind: "step", step: MAX_STEP });
    expect(scrubberKey("PageDown", 1)).toEqual({ kind: "step", step: 0 });
    expect(scrubberKey("Enter", 3)).toEqual({ kind: "ignore" });
  });
});

describe("playbackFrame", () => {
  it("advances six steps a second and stops at the last frame", () => {
    const quarter = playbackFrame(250, 0, MAX_STEP, true);
    expect(quarter.frame).toBeGreaterThanOrEqual(1);
    expect(quarter.frame).toBeLessThanOrEqual(2);
    expect(quarter.done).toBe(false);
    expect(playbackFrame(100_000, 0, MAX_STEP, true)).toEqual({
      frame: MAX_STEP,
      done: true,
    });
    expect(playbackFrame(0, MAX_STEP, MAX_STEP, true)).toEqual({
      frame: MAX_STEP,
      done: true,
    });
  });
});
