/**
 * The published scoreboard document and the table it drives.
 *
 * A cell carries its own test result, so the page never recomputes a badge.
 * A badge without an interval would claim a direction the data cannot support,
 * which is why the parser rejects it.
 */
export interface ScoreCell {
  readonly town: string;
  readonly variable: string;
  readonly unit: string;
  readonly lead_days: number;
  readonly model: string;
  readonly mae: number;
  readonly sample_size: number;
  readonly blocks: number;
  readonly statistic: number | null;
  readonly p_value: number | null;
  readonly mean_difference: number | null;
  readonly block_sd: number | null;
  readonly quarantined: boolean;
  readonly rejected: boolean;
  readonly interval: readonly [number, number] | null;
  readonly badge: string | null;
}

export interface Scoreboard {
  readonly generated_at: string;
  readonly stale_since: string;
  readonly window_days: number;
  readonly lead_days: readonly number[];
  readonly variables: readonly string[];
  readonly models: readonly string[];
  readonly towns: readonly string[];
  readonly cells: readonly ScoreCell[];
}

export interface ScoreboardRow {
  readonly town: string;
  readonly cells: ReadonlyMap<string, ScoreCell>;
}

export interface SortState {
  readonly key: string;
  readonly direction: "asc" | "desc";
}

export const DATA_FAULT_QUERY = "issues?q=label%3Adata-fault";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringField(value: Record<string, unknown>, name: string): string {
  const field = value[name];
  if (typeof field !== "string") {
    throw new Error(`scores.json field ${name} is not a string`);
  }
  return field;
}

function numberField(value: Record<string, unknown>, name: string): number {
  const field = value[name];
  if (typeof field !== "number" || !Number.isFinite(field)) {
    throw new Error(`scores.json field ${name} is not a finite number`);
  }
  return field;
}

function nullableNumber(value: unknown, name: string): number | null {
  if (value === null) {
    return null;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`scores.json field ${name} is not a finite number or null`);
  }
  return value;
}

function nullableString(value: unknown, name: string): string | null {
  if (value === null) {
    return null;
  }
  if (typeof value !== "string") {
    throw new Error(`scores.json field ${name} is not a string or null`);
  }
  return value;
}

function stringArray(value: unknown, name: string): readonly string[] {
  if (!Array.isArray(value) || value.some((item: unknown) => typeof item !== "string")) {
    throw new Error(`scores.json field ${name} is not a list of strings`);
  }
  return value as readonly string[];
}

function numberArray(value: unknown, name: string): readonly number[] {
  if (
    !Array.isArray(value) ||
    value.some((item: unknown) => typeof item !== "number" || !Number.isFinite(item))
  ) {
    throw new Error(`scores.json field ${name} is not a list of finite numbers`);
  }
  return value as readonly number[];
}

function parseInterval(value: unknown): readonly [number, number] | null {
  if (value === null) {
    return null;
  }
  if (
    !Array.isArray(value) ||
    value.length !== 2 ||
    typeof value[0] !== "number" ||
    typeof value[1] !== "number"
  ) {
    throw new Error("a scoreboard interval must be two numbers or null");
  }
  return [value[0], value[1]];
}

function parseCell(value: unknown): ScoreCell {
  if (!isRecord(value)) {
    throw new Error("scores.json contains a malformed cell");
  }
  const interval = parseInterval(value["interval"] ?? null);
  const badge = nullableString(value["badge"], "badge");
  if (badge !== null && interval === null) {
    throw new Error("a scoreboard badge must have an interval");
  }
  return {
    town: stringField(value, "town"),
    variable: stringField(value, "variable"),
    unit:
      typeof value["unit"] === "string"
        ? value["unit"]
        : ((
            { temperature_2m: "°C", pressure_msl: "hPa", wind_speed_10m: "m/s" } as Record<
              string,
              string
            >
          )[stringField(value, "variable")] ?? ""),
    lead_days: numberField(value, "lead_days"),
    model: stringField(value, "model"),
    mae: numberField(value, "mae"),
    sample_size: numberField(value, "sample_size"),
    blocks: numberField(value, "blocks"),
    statistic: nullableNumber(value["statistic"], "statistic"),
    p_value: nullableNumber(value["p_value"], "p_value"),
    mean_difference: nullableNumber(value["mean_difference"], "mean_difference"),
    block_sd: nullableNumber(value["block_sd"], "block_sd"),
    quarantined: value["quarantined"] === true,
    rejected: value["rejected"] === true,
    interval,
    badge,
  };
}

/** Read the published document, rejecting a badge the data cannot support. */
export function parseScoreboard(value: unknown): Scoreboard {
  if (!isRecord(value)) {
    throw new Error("scores.json is not an object");
  }
  const cells = value["cells"];
  if (!Array.isArray(cells)) {
    throw new Error("scores.json has no cells list");
  }
  return {
    generated_at: stringField(value, "generated_at"),
    stale_since: stringField(value, "stale_since"),
    window_days: numberField(value, "window_days"),
    lead_days: numberArray(value["lead_days"], "lead_days"),
    variables: stringArray(value["variables"], "variables"),
    models: stringArray(value["models"], "models"),
    towns: stringArray(value["towns"], "towns"),
    cells: cells.map(parseCell),
  };
}

/** One row per town, with the cells for the selected variable and lead. */
export function scoreboardRows(
  board: Scoreboard,
  variable: string,
  lead: number,
): readonly ScoreboardRow[] {
  const rows = new Map<string, Map<string, ScoreCell>>();
  for (const town of board.towns) {
    rows.set(town, new Map());
  }
  for (const cell of board.cells) {
    if (cell.variable !== variable || cell.lead_days !== lead) {
      continue;
    }
    rows.get(cell.town)?.set(cell.model, cell);
  }
  return board.towns.map((town) => ({ town, cells: rows.get(town) ?? new Map() }));
}

function compareRows(left: ScoreboardRow, right: ScoreboardRow, key: string): number {
  if (key === "town") {
    return left.town.localeCompare(right.town);
  }
  const leftCell = left.cells.get(key);
  const rightCell = right.cells.get(key);
  if (leftCell === undefined && rightCell === undefined) {
    return 0;
  }
  if (leftCell === undefined) {
    return 1;
  }
  if (rightCell === undefined) {
    return -1;
  }
  return leftCell.mae - rightCell.mae;
}

/** Sort a copy of the rows by town or by one model's mean absolute error. */
export function sortRows(
  rows: readonly ScoreboardRow[],
  state: SortState,
): readonly ScoreboardRow[] {
  const sorted = [...rows];
  sorted.sort((left, right) => {
    const comparison = compareRows(left, right, state.key);
    return state.direction === "asc" ? comparison : -comparison;
  });
  return sorted;
}

/** Toggle the direction when the same column is chosen again. */
export function nextSort(current: SortState, key: string): SortState {
  if (current.key === key) {
    return { key, direction: current.direction === "asc" ? "desc" : "asc" };
  }
  return { key, direction: "asc" };
}

/** The badge text, or null when the cell has no interval to back it. */
export function badgeText(cell: ScoreCell): string | null {
  return cell.interval === null ? null : cell.badge;
}

/** The repository's data-fault issue search, or a relative link without a repository. */
export function dataFaultHref(repository: string | null): string {
  return repository === null
    ? DATA_FAULT_QUERY
    : `https://github.com/${repository}/${DATA_FAULT_QUERY}`;
}

/** Whether the published document is older than its own backstop date. */
export function isStale(board: Scoreboard, today: string): boolean {
  return today > board.stale_since;
}

export function formatMae(cell: ScoreCell): string {
  return cell.mae.toFixed(2);
}

export function variableLabel(variable: string): string {
  const labels: Readonly<Record<string, string>> = {
    temperature_2m: "2 m temperature",
    pressure_msl: "Sea-level pressure",
    wind_speed_10m: "10 m wind speed",
  };
  return labels[variable] ?? variable.replaceAll("_", " ");
}

function sortHeader(label: string, key: string, sort: SortState): HTMLTableCellElement {
  const header = document.createElement("th");
  header.scope = "col";
  header.ariaSort =
    sort.key === key ? (sort.direction === "asc" ? "ascending" : "descending") : "none";
  const button = document.createElement("button");
  button.type = "button";
  button.dataset["sort"] = key;
  button.textContent = label;
  header.append(button);
  return header;
}

/** Draw the table; the header buttons are the keyboard-operable sort controls. */
export function renderScoreboardTable(
  table: HTMLTableElement,
  rows: readonly ScoreboardRow[],
  models: readonly string[],
  sort: SortState,
  faultHref: string,
): void {
  const head = table.tHead ?? table.createTHead();
  const body = table.tBodies[0] ?? table.createTBody();
  head.replaceChildren();
  body.replaceChildren();
  const header = document.createElement("tr");
  header.append(sortHeader("Town", "town", sort));
  for (const model of models) {
    header.append(sortHeader(model, model, sort));
  }
  head.append(header);
  for (const row of rows) {
    const tr = document.createElement("tr");
    const town = document.createElement("th");
    town.scope = "row";
    town.textContent = row.town;
    tr.append(town);
    for (const model of models) {
      const td = document.createElement("td");
      const cell = row.cells.get(model);
      if (cell === undefined) {
        td.textContent = "not enough data";
      } else {
        if (cell.quarantined) {
          const link = document.createElement("a");
          link.href = faultHref;
          link.textContent = "data fault";
          td.append(link);
        } else {
          td.textContent = formatMae(cell);
        }
        td.append(` ${cell.unit} · n=${String(cell.sample_size)}`);
        const badge = badgeText(cell);
        if (badge !== null && !cell.quarantined) {
          const span = document.createElement("span");
          span.className = "badge";
          span.textContent = ` ${badge}`;
          td.append(span);
        }
        if (cell.model === "WN2 mean" || cell.model === "WN2 member 0") {
          const detail = document.createElement("div");
          detail.className = "badge";
          detail.textContent = `WN2 − IFS interval: ${cell.interval === null ? "not estimated" : `${String(cell.interval[0])} to ${String(cell.interval[1])} ${cell.unit}`}`;
          td.append(detail);
        }
      }
      tr.append(td);
    }
    body.append(tr);
  }
}
