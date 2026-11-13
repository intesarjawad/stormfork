import { describe, expect, it } from "vitest";

import { EMPTY_VIEW, formatRoute, parseRoute, type Route } from "../src/route";

const FULL_VIEW = {
  step: 10,
  layer: "mslp",
  lead: 24,
  run: "fork",
  difference: false,
  stop: null,
  record: false,
};

describe("parseRoute", () => {
  it("reads the home route from an empty or bare hash", () => {
    expect(parseRoute("")).toEqual({ kind: "home" });
    expect(parseRoute("#/")).toEqual({ kind: "home" });
  });

  it("restores every experiment state field", () => {
    expect(parseRoute("#/e/sandy_sweep?step=10&layer=mslp&lead=24&run=fork")).toEqual({
      kind: "experiment",
      id: "sandy_sweep",
      view: {
        step: 10,
        layer: "mslp",
        lead: 24,
        run: "fork",
        difference: false,
        stop: null,
        record: false,
      },
    });
    expect(parseRoute("#/e/demo-fixture")).toEqual({
      kind: "experiment",
      id: "demo-fixture",
      view: EMPTY_VIEW,
    });
  });

  it("decodes an id and ignores non-numeric steps", () => {
    expect(parseRoute("#/e/storm%20name?step=abc&run=")).toMatchObject({
      id: "storm name",
      view: { step: null, run: null },
    });
  });

  it("reads the other published routes", () => {
    expect(parseRoute("#/how")).toEqual({ kind: "how" });
    expect(parseRoute("#/card")).toEqual({ kind: "card" });
    expect(parseRoute("#/scores/KJFK")).toEqual({ kind: "scores", icao: "KJFK" });
    expect(parseRoute("#/scores")).toEqual({ kind: "scores", icao: null });
  });

  it("marks a deep link as the recorded opening", () => {
    expect(parseRoute("#/e/sandy_sweep?record=1")).toEqual({
      kind: "experiment",
      id: "sandy_sweep",
      view: { ...EMPTY_VIEW, record: true },
    });
    expect(parseRoute("#/e/sandy_sweep?record=0")).toEqual({
      kind: "experiment",
      id: "sandy_sweep",
      view: EMPTY_VIEW,
    });
  });

  it("falls back to the home route for an unknown path", () => {
    expect(parseRoute("#/missing")).toEqual({ kind: "home" });
  });
});

describe("formatRoute", () => {
  it("writes the query in a stable order and leaves unset fields out", () => {
    expect(formatRoute({ kind: "experiment", id: "sandy_sweep", view: FULL_VIEW })).toBe(
      "#/e/sandy_sweep?step=10&layer=mslp&lead=24&run=fork",
    );
    expect(formatRoute({ kind: "experiment", id: "demo-fixture", view: EMPTY_VIEW })).toBe(
      "#/e/demo-fixture",
    );
    expect(
      formatRoute({
        kind: "experiment",
        id: "demo-fixture",
        view: { ...EMPTY_VIEW, record: true },
      }),
    ).toBe("#/e/demo-fixture?record=1");
  });

  it("round trips every route and state combination", () => {
    const routes: readonly Route[] = [
      { kind: "home" },
      { kind: "how" },
      { kind: "card" },
      { kind: "scores", icao: "KJFK" },
      { kind: "scores", icao: null },
      { kind: "experiment", id: "demo-fixture", view: EMPTY_VIEW },
      { kind: "experiment", id: "sandy_sweep", view: FULL_VIEW },
      {
        kind: "experiment",
        id: "storm name",
        view: { ...EMPTY_VIEW, step: 0, layer: "wind10", run: "baseline" },
      },
      {
        kind: "experiment",
        id: "milton",
        view: { ...EMPTY_VIEW, record: true, difference: true, stop: "east" },
      },
    ];
    for (const route of routes) {
      expect(parseRoute(formatRoute(route))).toEqual(route);
    }
  });
});
