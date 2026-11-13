import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  DATA_FAULT_QUERY,
  badgeText,
  dataFaultHref,
  formatMae,
  isStale,
  nextSort,
  parseScoreboard,
  scoreboardRows,
  sortRows,
  type Scoreboard,
} from "../src/scoreboard";

const FIXTURE = JSON.parse(
  readFileSync(new URL("./fixtures/scores.json", import.meta.url), "utf8"),
) as unknown;

function board(): Scoreboard {
  return parseScoreboard(FIXTURE);
}

describe("parseScoreboard", () => {
  it("reads the fixture and keeps every cell", () => {
    const parsed = board();
    expect(parsed.generated_at).toBe("2026-11-07");
    expect(parsed.towns).toHaveLength(24);
    expect(parsed.cells).toHaveLength(72);
  });

  it("retains the published unit and treats an absent interval as unestimated", () => {
    const input = structuredClone(FIXTURE) as {
      cells: { unit?: string; interval?: [number, number] | null }[];
    };
    const cell = input.cells[0];
    if (cell === undefined) throw new Error("fixture has no cells");
    cell.unit = "hPa";
    delete cell.interval;
    expect(parseScoreboard(input).cells[0]).toMatchObject({ unit: "hPa", interval: null });
  });

  it("rejects a badge without an interval", () => {
    const broken = structuredClone(FIXTURE) as {
      cells: { badge: string | null; interval: [number, number] | null }[];
    };
    const cell = broken.cells.find((candidate) => candidate.badge !== null);
    if (cell === undefined) {
      throw new Error("fixture has no badge");
    }
    cell.interval = null;
    expect(() => parseScoreboard(broken)).toThrow("badge must have an interval");
  });
});

describe("scoreboardRows", () => {
  it("builds one row per town for the selected variable and lead", () => {
    const rows = scoreboardRows(board(), "temperature_2m", 0);
    expect(rows).toHaveLength(24);
    expect(rows[0]?.town).toBe("KEYW");
    expect(rows[0]?.cells.size).toBe(3);
  });

  it("leaves a town without a cell for the selection empty", () => {
    const rows = scoreboardRows(board(), "pressure_msl", 0);
    expect(rows).toHaveLength(24);
    expect(rows.every((row) => row.cells.size === 0)).toBe(true);
  });
});

describe("sortRows", () => {
  it("sorts by town in both directions", () => {
    const rows = scoreboardRows(board(), "temperature_2m", 0);
    expect(sortRows(rows, { key: "town", direction: "asc" })[0]?.town).toBe("KACY");
    expect(sortRows(rows, { key: "town", direction: "desc" })[0]?.town).toBe("TJSJ");
  });

  it("sorts by a model's mean absolute error", () => {
    const rows = scoreboardRows(board(), "temperature_2m", 0);
    const sorted = sortRows(rows, { key: "WN2 member 0", direction: "asc" });
    expect(sorted[0]?.cells.get("WN2 member 0")?.mae).toBe(1.2);
  });

  it("toggles the direction only for the same column", () => {
    expect(nextSort({ key: "town", direction: "asc" }, "town")).toEqual({
      key: "town",
      direction: "desc",
    });
    expect(nextSort({ key: "town", direction: "desc" }, "IFS")).toEqual({
      key: "IFS",
      direction: "asc",
    });
  });
});

describe("badges and faults", () => {
  it("shows a badge only when the cell has an interval", () => {
    const cells = board().cells;
    const backed = cells.find((cell) => cell.badge !== null);
    const unbacked = cells.find((cell) => cell.badge === null && cell.interval === null);
    if (backed === undefined || unbacked === undefined) {
      throw new Error("fixture is missing a badge or an unbacked cell");
    }
    expect(badgeText(backed)).toBe("beats IFS");
    expect(badgeText(unbacked)).toBeNull();
  });

  it("links a quarantined cell to the data-fault search", () => {
    const quarantined = board().cells.find((cell) => cell.quarantined);
    expect(quarantined?.town).toBe("KJFK");
    expect(dataFaultHref("owner/repo").endsWith(DATA_FAULT_QUERY)).toBe(true);
    expect(dataFaultHref(null)).toBe(DATA_FAULT_QUERY);
  });

  it("formats a mean absolute error to two decimals", () => {
    const cell = board().cells[0];
    if (cell === undefined) {
      throw new Error("fixture has no cells");
    }
    expect(formatMae(cell)).toBe("1.00");
  });
});

describe("isStale", () => {
  it("compares today with the document's backstop date", () => {
    expect(isStale(board(), "2026-11-08")).toBe(false);
    expect(isStale(board(), "2026-11-10")).toBe(true);
  });
});
