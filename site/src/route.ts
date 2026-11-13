/**
 * Hash routes the site can restore from a URL.
 *
 * A view carries only the state the reader can name in a link: the step, the
 * field, the lead time and the run. Everything else is derived from the
 * experiment manifest. The record flag asks for the fixed recorded opening.
 */
export interface ExperimentView {
  readonly step: number | null;
  readonly layer: string | null;
  readonly lead: number | null;
  readonly run: string | null;
  readonly difference: boolean;
  readonly stop: string | null;
  readonly record: boolean;
}

export type Route =
  | { readonly kind: "home" }
  | { readonly kind: "experiment"; readonly id: string; readonly view: ExperimentView }
  | { readonly kind: "how" }
  | { readonly kind: "scores"; readonly icao: string | null }
  | { readonly kind: "card" };

export const EMPTY_VIEW: ExperimentView = {
  step: null,
  layer: null,
  lead: null,
  run: null,
  difference: false,
  stop: null,
  record: false,
};

function decode(segment: string): string {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}

function integer(value: string | null): number | null {
  if (value === null || !/^\d+$/.test(value)) {
    return null;
  }
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) ? parsed : null;
}

function pathAndQuery(hash: string): readonly [string, URLSearchParams] {
  const raw = hash.startsWith("#") ? hash.slice(1) : hash;
  const split = raw.indexOf("?");
  const path = split < 0 ? raw : raw.slice(0, split);
  return [path, new URLSearchParams(split < 0 ? "" : raw.slice(split + 1))];
}

function text(value: string | null): string | null {
  return value === null || value.length === 0 ? null : value;
}

/** Parse a `location.hash` value, tolerating a missing leading `#`. */
export function parseRoute(hash: string): Route {
  const [path, query] = pathAndQuery(hash);
  const segments = path.split("/").filter((segment) => segment.length > 0);
  const [head, second] = segments;
  if (head === undefined) {
    return { kind: "home" };
  }
  if (head === "how" && second === undefined) {
    return { kind: "how" };
  }
  if (head === "card" && second === undefined) {
    return { kind: "card" };
  }
  if (head === "scores") {
    return { kind: "scores", icao: second === undefined ? null : decode(second) };
  }
  if (head === "e" && second !== undefined) {
    return {
      kind: "experiment",
      id: decode(second),
      view: {
        step: integer(query.get("step")),
        layer: text(query.get("layer")),
        lead: integer(query.get("lead")),
        run: text(query.get("run")),
        difference: query.get("difference") === "1",
        stop: text(query.get("stop")),
        record: query.get("record") === "1",
      },
    };
  }
  return { kind: "home" };
}

/** Build the hash that restores a route, including only fields that are set. */
export function formatRoute(route: Route): string {
  if (route.kind === "home") {
    return "#/";
  }
  if (route.kind === "how") {
    return "#/how";
  }
  if (route.kind === "card") {
    return "#/card";
  }
  if (route.kind === "scores") {
    return route.icao === null ? "#/scores" : `#/scores/${encodeURIComponent(route.icao)}`;
  }
  const query = new URLSearchParams();
  if (route.view.step !== null) {
    query.set("step", String(route.view.step));
  }
  if (route.view.layer !== null) {
    query.set("layer", route.view.layer);
  }
  if (route.view.lead !== null) {
    query.set("lead", String(route.view.lead));
  }
  if (route.view.run !== null) {
    query.set("run", route.view.run);
  }
  if (route.view.difference) {
    query.set("difference", "1");
  }
  if (route.view.stop !== null) {
    query.set("stop", route.view.stop);
  }
  if (route.view.record) {
    query.set("record", "1");
  }
  const encoded = query.toString();
  const suffix = encoded.length > 0 ? `?${encoded}` : "";
  return `#/e/${encodeURIComponent(route.id)}${suffix}`;
}
