import { element, describe, option, renderState } from "./dom";
import {
  dataFaultHref,
  variableLabel,
  isStale,
  nextSort,
  parseScoreboard,
  renderScoreboardTable,
  scoreboardRows,
  sortRows,
  type Scoreboard,
  type SortState,
} from "./scoreboard";
const REPOSITORY = "intesarjawad/stormfork";
const SCOREBOARD_URL = "scores/scores.json";

export function createScoreboardRoute() {
  const summary = element("#summary", HTMLParagraphElement);
  let board: Scoreboard | null = null;
  let boardSort: SortState = { key: "town", direction: "asc" };
  let boardVariable = "";
  let boardLead = 0;
  let boardTown: string | null = null;
  const scoresRetry = element("#scores-retry", HTMLButtonElement);
  const scoreboardSection = element("#scoreboard", HTMLElement);
  const scoreboardStatus = element("#scoreboard-status", HTMLParagraphElement);
  const scoreboardVariableSelect = element("#scoreboard-variable", HTMLSelectElement);
  const scoreboardLeadSelect = element("#scoreboard-lead", HTMLSelectElement);
  const scoreboardTable = element("#scoreboard-table", HTMLTableElement);
  const scoreboardCaption = element("#scoreboard-caption", HTMLTableCaptionElement);
  const scoreboardFaultLink = element("#scoreboard-fault-link", HTMLAnchorElement);
  const scoresState = element("#scores-state", HTMLElement);

  function renderBoard(): void {
    if (board === null) {
      return;
    }
    const rows = sortRows(
      scoreboardRows(board, boardVariable, boardLead).filter(
        (row) => boardTown === null || row.town.toUpperCase() === boardTown,
      ),
      boardSort,
    );
    renderScoreboardTable(
      scoreboardTable,
      rows,
      board.models,
      boardSort,
      dataFaultHref(REPOSITORY),
    );
    scoreboardCaption.textContent = `Mean absolute error (MAE), lower is better. Scores generated ${board.generated_at}.`;
    if (isStale(board, new Date().toISOString().slice(0, 10))) {
      scoreboardCaption.textContent += ` Data is stale since ${board.stale_since}.`;
    }
  }

  async function openScoreboard(icao: string | null): Promise<void> {
    scoreboardSection.hidden = false;
    boardTown = icao?.toUpperCase() ?? null;
    summary.textContent = "Scoreboard.";
    scoreboardStatus.textContent = "Loading the scoreboard.";
    scoresRetry.hidden = true;
    let unpublished = false;
    try {
      const response = await fetch(new URL(SCOREBOARD_URL, document.baseURI));
      unpublished = response.status === 404;
      if (!response.ok) {
        throw new Error(`${SCOREBOARD_URL} responded ${String(response.status)}`);
      }
      const loaded = parseScoreboard(await response.json());
      board = loaded;
      boardVariable = loaded.variables[0] ?? "";
      boardLead = loaded.lead_days[0] ?? 0;
      scoreboardVariableSelect.replaceChildren(
        ...loaded.variables.map((variable) => option(variable, variableLabel(variable))),
      );
      scoreboardLeadSelect.replaceChildren(
        ...loaded.lead_days.map((lead) => option(String(lead), `${String(lead)} days`)),
      );
      scoreboardVariableSelect.value = boardVariable;
      scoreboardLeadSelect.value = String(boardLead);
      scoreboardFaultLink.href = dataFaultHref(REPOSITORY);
      scoresState.hidden = true;
      scoreboardStatus.textContent =
        boardTown === null
          ? ""
          : loaded.towns.includes(boardTown)
            ? `Showing ${boardTown} in the board.`
            : "Unknown town. Choose a station from the full scoreboard.";
      renderBoard();
    } catch (error) {
      board = null;
      scoreboardVariableSelect.replaceChildren();
      scoreboardLeadSelect.replaceChildren();
      scoresRetry.hidden = false;
      scoreboardStatus.textContent = "";
      renderState(scoresState, {
        kind: unpublished ? "empty" : "error",
        title: unpublished ? "No scores published yet" : "Scores could not be loaded",
        text: "Use Retry scores to try again. Maintainers can build the board from stored inputs.",
        command:
          "uv run stormfork scores build --wn2 <wn2-directory> --observations <observations.csv> --ifs <ifs.csv> --output site/public/scores/scores.json",
        detail: describe(error),
      });
      scoreboardTable.tHead?.replaceChildren();
      scoreboardTable.tBodies[0]?.replaceChildren();
    }
  }

  scoresRetry.addEventListener("click", () => {
    void openScoreboard(boardTown);
  });
  scoreboardVariableSelect.addEventListener("change", () => {
    boardVariable = scoreboardVariableSelect.value;
    renderBoard();
  });
  scoreboardLeadSelect.addEventListener("change", () => {
    boardLead = Number(scoreboardLeadSelect.value);
    renderBoard();
  });
  scoreboardTable.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLButtonElement)) {
      return;
    }
    const key = target.dataset["sort"];
    if (key === undefined) {
      return;
    }
    boardSort = nextSort(boardSort, key);
    renderBoard();
  });

  return {
    open: openScoreboard,
    close: () => {
      scoreboardSection.hidden = true;
    },
  };
}
