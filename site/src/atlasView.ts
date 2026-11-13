import type { AtlasState } from "./atlasState";
import { maxStep } from "./atlasState";
import { element, describe } from "./dom";
import { FIELD_LABELS, fieldLabel } from "./atlasPresentation";
import { SPEED_MAX_MPS } from "./fieldLayer";
import type { FieldLayer } from "./loadExperiment";
import { formatRoute, type Route } from "./route";
import { clampScrubberStep, playbackFrame, stepLabel } from "./scrubber";
import { drawSeparation, type DrawnChart } from "./separationChart";
import { stepTime } from "./steps";
import { divergenceText } from "./divergence";
import type { segmented } from "./segmented";

export function createAtlasView(
  state: AtlasState,
  reducedMotion: boolean,
  layerSegments: ReturnType<typeof segmented>,
) {
  const summary = element("#summary", HTMLParagraphElement);
  const frameInput = element("#frame", HTMLInputElement);
  const frameOutput = element("#frame-value", HTMLOutputElement);
  const sheetTime = element("#sheet-time", HTMLOutputElement);
  const playButton = element("#atlas-play", HTMLButtonElement);
  const fieldLegend = element("#field-legend", HTMLDivElement);
  const divergenceOutput = element("#divergence", HTMLParagraphElement);
  const sparklineSvg = element("#sparkline", SVGSVGElement);
  const layerSelect = element("#layer", HTMLSelectElement);
  const sweepInput = element("#displacement", HTMLInputElement);
  let atlasChart: DrawnChart | null = null;
  function timeLabel(frame: number): string {
    if (state.initTime === "") {
      return stepLabel(frame);
    }
    const { day, hour } = stepTime(state.initTime, frame);
    return `${day}, ${hour}`;
  }

  function validIso(frame: number): string {
    return new Date(Date.parse(`${state.initTime}Z`) + frame * 6 * 3_600_000).toISOString();
  }

  function stopPlayback(): void {
    if (state.playRequest !== null) {
      cancelAnimationFrame(state.playRequest);
      state.playRequest = null;
    }
    state.playing = false;
    playButton.textContent = "Play";
    playButton.ariaPressed = "false";
  }

  function syncRoute(): void {
    if (state.experiment === null) {
      return;
    }
    const route: Route = {
      kind: "experiment",
      id: state.experiment.manifest.id,
      view: {
        step: state.frame,
        layer: state.layer,
        lead: state.lead,
        run: state.run.length > 0 ? state.run : null,
        difference: state.difference,
        stop: sweepInput.dataset["stop"] ?? null,
        record: state.record,
      },
    };
    const hash = formatRoute(route);
    if (window.location.hash !== hash) {
      window.history.replaceState(null, "", hash);
    }
  }

  function updateReadout(): void {
    if (state.experiment === null) {
      return;
    }
    const field = state.experiment.manifest.fields.find((item) => item.name === state.layer);
    if (field !== undefined && state.globe?.field !== null) {
      const units: Readonly<Record<string, string>> = {
        mslp: "Pa",
        wind10: "m/s",
        t2m: "K",
        z500: "m²/s²",
        tp6h: "m per 6 h",
      };
      const unit = units[field.name] ?? "native units";
      const span = (field.range[1] - field.range[0]) / 2;
      const values = state.difference
        ? `${String(-span)} · 0 · +${String(span)}`
        : field.name === "wind10"
          ? `0 · ${String(SPEED_MAX_MPS)}`
          : `${String(field.range[0])} · ${String(field.range[1])}`;
      fieldLegend.textContent = `${fieldLabel(field)}${state.difference ? " difference from baseline" : ""}: ${values} ${unit}${state.difference ? " (ends saturate)" : ""}`;
      fieldLegend.dataset["difference"] = String(state.difference);
      fieldLegend.hidden = false;
    }
    const difference = state.difference ? ", difference from baseline" : "";
    summary.textContent =
      state.globe?.field === null
        ? `${state.experiment.manifest.title}: no field is available.`
        : `${state.experiment.manifest.title}: ${FIELD_LABELS[state.layer] ?? state.layer}, ${timeLabel(state.frame)}${difference}.`;
    if (state.fieldProblem !== "") summary.textContent += ` ${state.fieldProblem}`;
    if (state.trackProblem !== null) {
      summary.textContent += ` Tracks could not be loaded: ${describe(state.trackProblem)}`;
    }
    const point = state.divergence.find((entry) => entry.step === state.frame) ?? null;
    divergenceOutput.textContent = divergenceText(point);
    sparklineSvg.toggleAttribute("hidden", state.divergence.length === 0);
    if (state.divergence.length > 0) {
      const last = Math.max(...state.divergence.map((entry) => entry.step));
      atlasChart = drawSeparation(
        sparklineSvg,
        state.divergence.map((entry) => ({
          step: entry.step,
          validTime: "",
          forkKm: entry.forkKm,
          controlKm: entry.controlKm,
          pairedMembers: 0,
          pairedP: null,
        })),
        { width: 285, height: 96, compact: true, lastStep: last, initTime: state.initTime },
      );
      atlasChart.reveal(Math.min(state.frame, last));
    }
  }

  function setFrame(frame: number, sync: boolean): void {
    state.frame = clampScrubberStep(frame, maxStep(state));
    frameInput.value = String(state.frame);
    frameOutput.value = timeLabel(state.frame);
    sheetTime.value = timeLabel(state.frame);
    state.globe?.setFrame(state.frame);
    if (state.initTime !== "") {
      state.globe?.setSun(validIso(state.frame));
    }
    updateReadout();
    if (sync) {
      syncRoute();
    }
  }

  function setPlaying(playing: boolean): void {
    if (playing === state.playing) {
      return;
    }
    if (!playing) {
      stopPlayback();
      syncRoute();
      return;
    }
    if (state.frame >= maxStep(state)) {
      setFrame(0, false);
    }
    state.playing = true;
    state.playFrom = state.frame;
    state.playStart = performance.now();
    playButton.textContent = "Pause";
    playButton.ariaPressed = "true";
    state.playRequest = requestAnimationFrame(tickPlayback);
  }

  function tickPlayback(now: number): void {
    if (!state.playing) {
      return;
    }
    const next = playbackFrame(
      now - state.playStart,
      state.playFrom,
      maxStep(state),
      reducedMotion,
    );
    setFrame(next.frame, false);
    if (next.done) {
      setPlaying(false);
      return;
    }
    state.playRequest = requestAnimationFrame(tickPlayback);
  }

  async function showField(name: string, sync: boolean): Promise<void> {
    const loaded = state.experiment;
    if (loaded === null) {
      return;
    }
    const field = loaded.manifest.fields.find((candidate) => candidate.name === name);
    if (field === undefined) {
      summary.textContent = `${name} is not published by this experiment.`;
      return;
    }
    const ticket = ++state.fieldTicket;
    summary.textContent = `Loading ${fieldLabel(field)}.`;
    let layer: FieldLayer;
    try {
      layer = await loaded.loadField(name);
    } catch (error) {
      layer = { kind: "unavailable", field, error };
    }
    if (ticket !== state.fieldTicket) {
      return;
    }
    if (layer.kind !== "ready") {
      layerSelect.value = state.layer;
      layerSegments.sync();
      state.fieldProblem = `${FIELD_LABELS[name] ?? name} could not be loaded${state.globe?.field === null ? "." : "; the previous field remains visible."}`;
      updateReadout();
      if (sync) {
        syncRoute();
      }
      return;
    }
    state.fieldProblem = "";
    state.layer = name;
    layerSelect.value = name;
    layerSegments.sync();
    state.globe?.setField(layer.field, layer.textures, state.run, state.difference);
    state.globe?.setFrame(state.frame);
    updateReadout();
    if (sync) {
      syncRoute();
    }
  }
  return { timeLabel, stopPlayback, syncRoute, updateReadout, setFrame, setPlaying, showField };
}
