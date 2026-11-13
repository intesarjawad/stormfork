import { describe, expect, it } from "vitest";

import type { SweepStop } from "../src/manifest";
import { sweepStopLabel } from "../src/sweep";

function stop(direction: SweepStop["direction"], distance_km: number): SweepStop {
  return {
    id: `${direction}_${String(distance_km)}`,
    direction,
    distance_km,
    tracks: `stops/${direction}_${String(distance_km)}/tracks.json`,
  };
}

describe("sweepStopLabel", () => {
  it("states the distance and direction", () => {
    expect(sweepStopLabel(stop("east", 300))).toBe("Storm moved 300 km east");
    expect(sweepStopLabel(stop("west", 150))).toBe("Storm moved 150 km west");
    expect(sweepStopLabel(stop("north", 75))).toBe("Storm moved 75 km north");
    expect(sweepStopLabel(stop("south", 150))).toBe("Storm moved 150 km south");
  });

  it("names the zero edit as the null test", () => {
    expect(sweepStopLabel(stop("none", 0))).toBe("not moved, null test");
    expect(sweepStopLabel(stop("east", 0))).toBe("not moved, null test");
  });
});
