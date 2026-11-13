import { describe, expect, it } from "vitest";

import {
  RECORD_DRAG_MS,
  RECORD_HOLD_MS,
  RECORD_PLAY_MS,
  RECORD_STILL_MS,
  RECORD_TOTAL_MS,
  recordFrame,
} from "../src/record";

describe("recordFrame", () => {
  it("holds the signature still at frame 0 while it fades", () => {
    expect(recordFrame(0, 3, 40)).toEqual({
      phase: "still",
      progress: 0,
      stop: 0,
      step: 0,
      stillOpacity: 1,
      done: false,
    });
    expect(recordFrame(750, 3, 40)).toEqual({
      phase: "still",
      progress: 0.5,
      stop: 0,
      step: 0,
      stillOpacity: 0.5,
      done: false,
    });
  });

  it("drags the displacement slider from 0 to the last stop", () => {
    expect(recordFrame(RECORD_STILL_MS, 3, 40)).toMatchObject({
      phase: "drag",
      progress: 0,
      stop: 0,
      step: 0,
      stillOpacity: 0,
    });
    expect(recordFrame(RECORD_STILL_MS + RECORD_DRAG_MS / 2, 3, 40).stop).toBe(1);
    expect(recordFrame(RECORD_STILL_MS + RECORD_DRAG_MS - 1, 3, 40).stop).toBe(2);
  });

  it("plays steps 0 to 40 over four seconds", () => {
    const playStart = RECORD_STILL_MS + RECORD_DRAG_MS;
    expect(recordFrame(playStart, 3, 40)).toMatchObject({
      phase: "play",
      progress: 0,
      stop: 2,
      step: 0,
    });
    expect(recordFrame(playStart + RECORD_PLAY_MS / 2, 3, 40).step).toBe(20);
    expect(recordFrame(playStart + RECORD_PLAY_MS - 1, 3, 40).step).toBe(40);
  });

  it("freezes on the last frame for two seconds", () => {
    const holdStart = RECORD_STILL_MS + RECORD_DRAG_MS + RECORD_PLAY_MS;
    expect(recordFrame(holdStart, 3, 40)).toMatchObject({
      phase: "hold",
      progress: 0,
      stop: 2,
      step: 40,
      done: false,
    });
    expect(recordFrame(RECORD_TOTAL_MS - 1, 3, 40).done).toBe(false);
    expect(recordFrame(RECORD_TOTAL_MS, 3, 40).done).toBe(true);
  });

  it("keeps the total at ten seconds and tolerates a missing sweep", () => {
    expect(RECORD_STILL_MS + RECORD_DRAG_MS + RECORD_PLAY_MS + RECORD_HOLD_MS).toBe(10_000);
    expect(recordFrame(RECORD_STILL_MS + 1, 0, 40).stop).toBe(0);
    expect(recordFrame(RECORD_STILL_MS + RECORD_DRAG_MS + 1, 1, 0)).toMatchObject({
      stop: 0,
      step: 0,
    });
  });

  it("clamps negative elapsed time to the start", () => {
    expect(recordFrame(-500, 3, 40)).toMatchObject({
      phase: "still",
      progress: 0,
      stillOpacity: 1,
    });
  });
});
