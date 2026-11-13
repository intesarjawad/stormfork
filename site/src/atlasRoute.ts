import { BACKGROUND_COLOUR } from "./colours";
import { element, describe, option, renderState } from "./dom";
import { GlobeRenderer } from "./globeRenderer";
import { DEFAULT_FIELD, loadExperiment, type Experiment } from "./loadExperiment";
import { leadTimes, loadStopDivergence, loadReproductionText } from "./experimentData";
import {
  RUN_LABELS,
  fieldLabel,
  renderLegend,
  webGlAvailable,
  selectedField,
  selectedRun,
} from "./atlasPresentation";
import { createAtlasState, maxStep } from "./atlasState";
import { createAtlasView } from "./atlasView";
import { createAtlasSweep } from "./atlasSweep";
import { createAtlasRecording } from "./atlasRecording";
import { focusBasin, openingMove, markFirstFrame } from "./atlasCamera";
import { probeField } from "./probe";
import type { ExperimentView } from "./route";
import { MAX_STEP, scrubberKey } from "./scrubber";
import { segmented } from "./segmented";
import { loadTracks } from "./tracks";
import { seriesFromTracks } from "./trackSeries";
declare global {
  interface Window {
    stormforkGlobe?: GlobeRenderer | undefined;
    stormforkProbe?:
      | ((frame: number, lat: number, lon: number) => readonly [number, number, number, number])
      | undefined;
  }
}
const HONESTY_KEY = "stormfork:honesty-seen";
const FAILURE_WORDS = /could not|unavailable|not published|not found/i;

export function createAtlasRoute() {
  const summary = element("#summary", HTMLParagraphElement);
  const panel = element("#globe-panel", HTMLDivElement);
  const form = element("#controls", HTMLFormElement);
  const layerSelect = element("#layer", HTMLSelectElement);
  const runSelect = element("#run", HTMLSelectElement);
  const leadField = element("#lead-field", HTMLParagraphElement);
  const leadSelect = element("#lead", HTMLSelectElement);
  const frameInput = element("#frame", HTMLInputElement);
  const frameOutput = element("#frame-value", HTMLOutputElement);
  const playButton = element("#atlas-play", HTMLButtonElement);
  const differenceInput = element("#difference", HTMLInputElement);
  const displacementField = element("#displacement-field", HTMLFieldSetElement);
  const sweepInput = element("#displacement", HTMLInputElement);
  const sparklineSvg = element("#sparkline", SVGSVGElement);
  const fallback = element("#fallback", HTMLDivElement);
  const fallbackError = element("#fallback-error", HTMLSpanElement);
  const stageState = element("#stage-state", HTMLElement);
  const sheetToggle = element("#sheet-toggle", HTMLButtonElement);
  const honesty = element("#honesty", HTMLElement);
  const honestyDismiss = element("#honesty-dismiss", HTMLButtonElement);
  const noticeLink = element("#notice-link", HTMLParagraphElement);
  const legend = element("#legend", HTMLUListElement);
  const tracksSection = element("#tracks-section", HTMLElement);
  const reproductions = element("#reproductions", HTMLParagraphElement);
  const fieldLegend = element("#field-legend", HTMLDivElement);
  const state = createAtlasState();
  const layerSegments = segmented(layerSelect);
  const runSegments = segmented(runSelect);
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const view = createAtlasView(state, reducedMotion, layerSegments);
  const { timeLabel, stopPlayback, syncRoute, updateReadout, setFrame, setPlaying, showField } =
    view;
  const sweep = createAtlasSweep(state, view);
  const { selectStop, configureSweep, showSweepChart } = sweep;
  const recording = createAtlasRecording(state, panel, setFrame, selectStop);
  const startRecordMode = recording.start;
  const exitRecordMode = recording.close;
  let resizeObserver: ResizeObserver | null = null;
  let firstFrameMarked = false;
  panel.style.backgroundColor = BACKGROUND_COLOUR;
  frameInput.max = String(MAX_STEP);
  frameInput.value = "0";
  frameOutput.value = timeLabel(0);
  const fields = element("#control-fields", HTMLDivElement);
  const mobile = window.matchMedia("(max-width: 959px)");
  const updateSheet = (): void => {
    fields.inert = mobile.matches && form.dataset["open"] !== "true";
  };
  mobile.addEventListener("change", updateSheet);
  updateSheet();
  element(".skip", HTMLAnchorElement).addEventListener("click", (event) => {
    event.preventDefault();
    element("#main", HTMLElement).focus();
  });

  new MutationObserver(() => {
    const failed = FAILURE_WORDS.test(summary.textContent);
    summary.dataset["tone"] = failed ? "error" : "info";
    noticeLink.hidden = !failed || document.documentElement.dataset["route"] !== "atlas";
  }).observe(summary, { childList: true, characterData: true, subtree: true });

  function showHonesty(): void {
    try {
      honesty.hidden = window.localStorage.getItem(HONESTY_KEY) === "1";
    } catch {
      honesty.hidden = false;
    }
  }

  sheetToggle.addEventListener("click", () => {
    const open = form.dataset["open"] !== "true";
    form.dataset["open"] = String(open);
    sheetToggle.ariaExpanded = String(open);
    updateSheet();
  });

  honestyDismiss.addEventListener("click", () => {
    honesty.hidden = true;
    try {
      window.localStorage.setItem(HONESTY_KEY, "1");
    } catch {
      // Private windows refuse storage; the card then returns on the next visit.
    }
  });

  function applyView(view: ExperimentView): void {
    const loaded = state.experiment;
    if (loaded === null) {
      return;
    }
    stopPlayback();
    state.difference = view.difference;
    differenceInput.checked = state.difference;
    state.globe?.setDifference(state.difference);
    const stopIndex = state.sweep?.stops.findIndex((stop) => stop.id === view.stop) ?? -1;
    if (stopIndex >= 0) void selectStop(stopIndex, false);
    state.record = view.record;
    const fields = loaded.manifest.fields;
    const layer = selectedField(fields, view.layer);
    if (layer !== state.layer) {
      void showField(layer, false);
    } else {
      state.layer = layer;
      layerSelect.value = layer;
      layerSegments.sync();
    }

    state.run = selectedRun(loaded.runs, view.run);
    runSelect.value = state.run;
    runSegments.sync();
    state.globe?.setRun(state.run);

    const leads = leadTimes(loaded.manifest);
    state.lead = view.lead;
    if (state.lead !== null && leads.includes(state.lead)) {
      leadSelect.value = String(state.lead);
    }
    setFrame(view.step ?? 0, false);
    if (state.record && state.globe !== null) {
      startRecordMode(state.globe);
    } else {
      exitRecordMode();
    }
  }

  function closeExperiment(): void {
    exitRecordMode();
    state.openTicket += 1;
    state.fieldTicket += 1;
    state.stopTicket += 1;
    stopPlayback();
    resizeObserver?.disconnect();
    resizeObserver = null;
    state.globe?.dispose();
    state.globe = null;
    window.stormforkGlobe = undefined;
    window.stormforkProbe = undefined;
    state.experiment = null;
    state.id = "";
    state.tracks = [];
    state.trackProblem = null;
    state.fieldProblem = "";
    state.sweep = null;
    state.divergence = [];
    state.stopTracks.clear();
    sweep.close();
    panel.replaceChildren();
    panel.hidden = true;
    honesty.hidden = true;
    form.hidden = true;
    fallback.hidden = true;
    delete document.documentElement.dataset["renderer"];
    tracksSection.hidden = true;
    reproductions.hidden = true;
    reproductions.replaceChildren();
    legend.replaceChildren();
    fieldLegend.hidden = true;
    layerSelect.replaceChildren();
    runSelect.replaceChildren();
    leadSelect.replaceChildren();
    leadField.hidden = true;
    displacementField.hidden = true;
    sparklineSvg.toggleAttribute("hidden", true);
  }

  async function openExperiment(id: string, view: ExperimentView): Promise<void> {
    closeExperiment();
    const ticket = state.openTicket;
    state.id = id;
    state.record = view.record;
    state.layer = view.layer !== null ? view.layer : DEFAULT_FIELD;
    state.frame = view.step ?? 0;
    summary.textContent = `Loading ${id}.`;
    renderState(stageState, {
      kind: "loading",
      title: "Reading the run",
      text: "The tracks come first, then the pressure atlas.",
    });

    let loaded: Experiment;
    try {
      loaded = await loadExperiment(id);
    } catch (error) {
      if (ticket === state.openTicket) {
        summary.textContent = `The atlas for ${id} is not available.`;
        renderState(stageState, {
          kind: "error",
          title: "This atlas is not in the build",
          text: `The files for ${id} are fetched from the experiment's release, not committed.`,
          command: "uv run python scripts/fetch_atlases.py",
          detail: describe(error),
        });
      }
      return;
    }
    if (ticket !== state.openTicket) {
      return;
    }
    state.experiment = loaded;
    state.initTime = loaded.manifest.init?.time ?? "";
    frameInput.max = String(maxStep(state));
    state.difference = view.difference;
    differenceInput.checked = state.difference;

    for (const field of loaded.manifest.fields) {
      layerSelect.append(option(field.name, fieldLabel(field)));
    }
    for (const run of loaded.runs) {
      runSelect.append(option(run, RUN_LABELS[run] ?? run));
    }
    const leads = leadTimes(loaded.manifest);
    for (const lead of leads) {
      leadSelect.append(option(String(lead), `${String(lead)} h`));
    }
    leadField.hidden = leads.length === 0;

    const fields = loaded.manifest.fields;
    state.layer = selectedField(fields, view.layer);
    layerSelect.value = state.layer;
    layerSegments.sync();

    state.run = selectedRun(loaded.runs, view.run);
    runSelect.value = state.run;
    runSegments.sync();

    state.lead = view.lead;
    if (state.lead !== null && leads.includes(state.lead)) {
      leadSelect.value = String(state.lead);
    }

    state.sweep = loaded.manifest.sweep ?? null;
    configureSweep(state.sweep);

    try {
      state.tracks = await loadTracks(id);
    } catch (error) {
      state.trackProblem = error;
    }
    state.divergence = await loadStopDivergence(id, "divergence.json");
    const reproduced = await loadReproductionText(id);
    if (ticket !== state.openTicket) {
      return;
    }
    reproductions.textContent = reproduced;
    reproductions.hidden = reproduced === null;

    legend.hidden = false;
    renderLegend(legend, loaded.runs);
    tracksSection.hidden = state.tracks.length === 0;
    form.hidden = false;
    if (state.sweep !== null) {
      void showSweepChart(state.sweep, ticket);
    }

    if (!webGlAvailable()) {
      document.documentElement.dataset["renderer"] = "fallback";
      panel.hidden = true;
      form.hidden = true;
      fallback.hidden = false;
      stageState.hidden = true;
      legend.hidden = true;
      fallbackError.textContent =
        state.trackProblem === null
          ? ""
          : ` Tracks could not be loaded: ${describe(state.trackProblem)}`;
      summary.textContent =
        state.trackProblem === null
          ? "WebGL is unavailable; the globe cannot be displayed."
          : `WebGL is unavailable and tracks could not be loaded: ${describe(state.trackProblem)}`;
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.tabIndex = 0;
    canvas.role = "img";
    canvas.ariaLabel = "Globe: arrow keys rotate; plus and minus zoom";
    panel.replaceChildren(canvas);
    panel.hidden = false;
    showHonesty();

    let renderer: GlobeRenderer;
    try {
      renderer = new GlobeRenderer(canvas, { reducedMotion });
    } catch (error) {
      if (ticket !== state.openTicket) {
        return;
      }
      panel.hidden = true;
      form.hidden = true;
      fallback.hidden = false;
      stageState.hidden = true;
      legend.hidden = true;
      summary.textContent = `The globe could not start: ${describe(error)}`;
      fallbackError.textContent = ` ${summary.textContent}`;
      document.documentElement.dataset["renderer"] = "fallback";
      return;
    }
    if (ticket !== state.openTicket) {
      renderer.dispose();
      return;
    }
    state.globe = renderer;
    stageState.hidden = true;
    document.documentElement.dataset["renderer"] = "webgl";
    window.stormforkGlobe = renderer;
    window.stormforkProbe = (frame, lat, lon) => probeField(renderer, frame, lat, lon);
    renderer.setTracks(seriesFromTracks(state.tracks));

    resizeObserver = new ResizeObserver(() => {
      renderer.resize(panel.clientWidth, panel.clientHeight);
    });
    resizeObserver.observe(panel);

    await showField(state.layer, false);
    const initialStop = state.sweep?.stops.findIndex((stop) => stop.id === view.stop) ?? -1;
    if (initialStop >= 0) await selectStop(initialStop, false);
    if (ticket !== state.openTicket) {
      return;
    }
    setFrame(view.step ?? 0, true);
    focusBasin(renderer, panel, state.tracks);
    openingMove(renderer, canvas, state, reducedMotion);
    if (!firstFrameMarked) {
      firstFrameMarked = true;
      markFirstFrame();
    }
    if (view.record) {
      startRecordMode(renderer);
    }
  }
  layerSelect.addEventListener("change", () => {
    void showField(layerSelect.value, true);
  });
  runSelect.addEventListener("change", () => {
    state.run = runSelect.value;
    state.globe?.setRun(state.run);
    updateReadout();
    syncRoute();
  });
  leadSelect.addEventListener("change", () => {
    const lead = Number(leadSelect.value);
    state.lead = Number.isFinite(lead) ? lead : null;
    if (state.lead !== null) {
      setFrame(Math.round(state.lead / 6), true);
    } else {
      syncRoute();
    }
  });
  differenceInput.addEventListener("change", () => {
    state.difference = differenceInput.checked;
    state.globe?.setDifference(state.difference);
    updateReadout();
    syncRoute();
  });
  playButton.addEventListener("click", () => {
    setPlaying(!state.playing);
  });
  frameInput.addEventListener("input", () => {
    setPlaying(false);
    setFrame(Number(frameInput.value), true);
  });
  frameInput.addEventListener("keydown", (event) => {
    const action = scrubberKey(event.key, state.frame, maxStep(state));
    if (action.kind === "ignore") {
      return;
    }
    event.preventDefault();
    if (action.kind === "toggle") {
      setPlaying(!state.playing);
    } else {
      setPlaying(false);
      setFrame(action.step, true);
    }
  });
  sweepInput.addEventListener("input", () => {
    void selectStop(Number(sweepInput.value), true);
  });

  return {
    open: openExperiment,
    applyView,
    close: closeExperiment,
    id: () => state.experiment?.manifest.id,
  };
}
