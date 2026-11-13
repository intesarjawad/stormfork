import { describe, expect, it } from "vitest";

import type { SweepStop } from "../src/manifest";
import {
  memberLatitudes,
  signedOffsetKm,
  sweepChartLayout,
  sweepColumn,
  sweepColumnText,
} from "../src/sweepChart";
import type { TrackSample } from "../src/tracks";

function stop(direction: SweepStop["direction"], distance_km: number): SweepStop {
  const id = `${direction}_${String(distance_km)}`;
  return { id, direction, distance_km, tracks: `stops/${id}/tracks.json` };
}

function sample(run: string, member: number, step: number, lat: number): TrackSample {
  return { run, member, step, lat, lon: -75 };
}

const SAMPLES: readonly TrackSample[] = [
  sample("fork", 1, 0, 30),
  sample("fork", 1, 4, 36),
  sample("fork", 0, 0, 30),
  sample("fork", 0, 4, 34),
  sample("fork", 2, 0, 30),
  sample("control", 0, 4, 35),
  sample("control", 1, 4, 37),
  sample("baseline", 0, 4, 99),
];

describe("memberLatitudes", () => {
  it("reads each member of one run at the step, in member order", () => {
    expect(memberLatitudes(SAMPLES, "fork", 4)).toEqual([
      { member: 0, lat: 34 },
      { member: 1, lat: 36 },
    ]);
  });

  it("uses each member's last sample when no step is given", () => {
    expect(memberLatitudes(SAMPLES, "fork", null)).toEqual([
      { member: 0, lat: 34 },
      { member: 1, lat: 36 },
      { member: 2, lat: 30 },
    ]);
  });

  it("returns nothing for a run that is absent", () => {
    expect(memberLatitudes(SAMPLES, "observed", 4)).toEqual([]);
  });
});

describe("sweepColumn", () => {
  it("signs the displacement and takes the control range at the step", () => {
    expect(signedOffsetKm(stop("west", 300))).toBe(-300);
    expect(signedOffsetKm(stop("south", 150))).toBe(-150);
    expect(signedOffsetKm(stop("east", 150))).toBe(150);
    expect(signedOffsetKm(stop("north", 75))).toBe(75);
    expect(signedOffsetKm(stop("none", 0))).toBe(0);
    const column = sweepColumn(stop("east", 150), SAMPLES, 4);
    expect(column.offsetKm).toBe(150);
    expect(column.members.map((entry) => entry.lat)).toEqual([34, 36]);
    expect(column.control).toEqual([35, 37]);
  });

  it("has no control range when the run is not published", () => {
    expect(sweepColumn(stop("none", 0), SAMPLES.slice(0, 5), 4).control).toBeNull();
  });
});

describe("sweepChartLayout", () => {
  const columns = [
    sweepColumn(stop("east", 300), SAMPLES, 4),
    sweepColumn(stop("west", 300), SAMPLES, 4),
    sweepColumn(stop("none", 0), SAMPLES, 4),
  ];

  it("places one point per member in each column, ordered west to east", () => {
    const layout = sweepChartLayout(columns, 400, 200);
    expect(layout.points).toHaveLength(6);
    expect(layout.xTicks.map((tick) => tick.label)).toEqual(["-300", "0", "300"]);
    const [first, middle, last] = layout.xTicks;
    expect(first?.position).toBe(layout.plot.left);
    expect(last?.position).toBe(layout.plot.right);
    expect(middle?.position).toBe((layout.plot.left + layout.plot.right) / 2);
    expect(layout.points.filter((point) => point.stop === "none_0")).toHaveLength(2);
  });

  it("puts higher latitudes nearer the top", () => {
    const layout = sweepChartLayout(columns, 400, 200);
    const north = layout.points.find((point) => point.lat === 36);
    const south = layout.points.find((point) => point.lat === 34);
    expect(north?.y).toBeLessThan(south?.y ?? 0);
    for (const point of layout.points) {
      expect(point.y).toBeGreaterThan(layout.plot.top);
      expect(point.y).toBeLessThan(layout.plot.bottom);
    }
  });

  it("maps the data range onto the plot with five percent padding", () => {
    // Latitudes span 34 to 37, so the axis runs 33.85 to 37.15.
    const layout = sweepChartLayout(columns.slice(0, 1), 400, 200);
    const height = layout.plot.bottom - layout.plot.top;
    const top = layout.points.find((point) => point.lat === 36);
    expect(top?.y).toBeCloseTo(layout.plot.top + ((37.15 - 36) / 3.3) * height, 6);
  });

  it("draws the control band as a closed polygon over every stop with a control", () => {
    const layout = sweepChartLayout(columns, 400, 200);
    expect(layout.band.split(" ")).toHaveLength(6);
    expect(sweepChartLayout(columns.slice(0, 1), 400, 200).band).toBe("");
  });

  it("centres a single column and survives a flat latitude", () => {
    const flat = sweepColumn(stop("none", 0), [sample("fork", 0, 4, 30)], 4);
    const layout = sweepChartLayout([flat], 400, 200);
    expect(layout.points[0]?.x).toBe((layout.plot.left + layout.plot.right) / 2);
    expect(layout.points[0]?.y).toBeCloseTo((layout.plot.top + layout.plot.bottom) / 2, 6);
    expect(layout.yTicks).toHaveLength(1);
  });

  it("handles an empty sweep", () => {
    const layout = sweepChartLayout([], 400, 200);
    expect(layout.points).toEqual([]);
    expect(layout.xTicks).toEqual([]);
    expect(layout.yTicks).toEqual([]);
  });
});

describe("sweepColumnText", () => {
  it("states the member range beside the control spread", () => {
    expect(
      sweepColumnText(sweepColumn(stop("east", 150), SAMPLES, 4), "Storm moved 150 km east"),
    ).toBe(
      "Storm moved 150 km east: landfall latitude 34.0 to 36.0 °N across 2 members; control spread 35.0 to 37.0 °N.",
    );
  });

  it("says so when the control or the members are missing", () => {
    const noControl = sweepColumn(stop("none", 0), SAMPLES.slice(0, 5), 4);
    expect(sweepColumnText(noControl, "not moved")).toBe(
      "not moved: landfall latitude 34.0 to 36.0 °N across 2 members; no control spread is published.",
    );
    const none = sweepColumn(stop("none", 0), [], 4);
    expect(sweepColumnText(none, "not moved")).toBe(
      "not moved: no member has a position at the landfall step.",
    );
  });
});
