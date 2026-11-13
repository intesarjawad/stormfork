import type { AtlasState } from "./atlasState";
import type { createAtlasView } from "./atlasView";
import { element, describe } from "./dom";
import { loadStopTracks, loadStopDivergence } from "./experimentData";
import type { Sweep, SweepStop } from "./manifest";
import { seriesFromTracks } from "./trackSeries";
import type { TrackSample } from "./tracks";
import { sweepStopLabel } from "./sweep";
import { sweepChartLayout, sweepColumn, sweepColumnText, type SweepColumn } from "./sweepChart";
import { CHART_POINT_RADIUS, CHART_SELECTED_RADIUS, drawSweepChart } from "./sweepPlot";

export function createAtlasSweep(state: AtlasState, view: ReturnType<typeof createAtlasView>) {
  const sweepChart = element("#sweep-chart", SVGSVGElement);
  const sweepChartReadout = element("#sweep-chart-readout", HTMLParagraphElement);
  const sweepInput = element("#displacement", HTMLInputElement);
  const sweepOutput = element("#displacement-value", HTMLOutputElement);
  const summary = element("#summary", HTMLParagraphElement);
  const sweepChartSection = element("#sweep-chart-section", HTMLElement);
  const displacementField = element("#displacement-field", HTMLFieldSetElement);
  const { updateReadout, syncRoute } = view;
  let chartColumns: readonly SweepColumn[] = [];
  function markChartStop(stop: SweepStop): void {
    for (const column of sweepChart.querySelectorAll<SVGGElement>("g[data-stop]")) {
      const selected = column.dataset["stop"] === stop.id;
      column.dataset["selected"] = String(selected);
      for (const circle of column.querySelectorAll("circle")) {
        circle.setAttribute("r", String(selected ? CHART_SELECTED_RADIUS : CHART_POINT_RADIUS));
      }
    }
    const current = chartColumns.find((column) => column.stop.id === stop.id);
    if (current !== undefined) {
      sweepChartReadout.textContent = sweepColumnText(current, sweepStopLabel(stop));
    }
  }

  function setSweepLabel(stop: SweepStop): void {
    const label = sweepStopLabel(stop);
    sweepOutput.value = label;
    sweepInput.ariaValueText = label;
    sweepInput.dataset["stop"] = stop.id;
    markChartStop(stop);
  }

  function trackForStop(index: number, stop: SweepStop): Promise<readonly TrackSample[]> {
    const cached = state.stopTracks.get(index);
    if (cached !== undefined) {
      return Promise.resolve(cached);
    }
    return loadStopTracks(state.id, stop).then((tracks) => {
      state.stopTracks.set(index, tracks);
      return tracks;
    });
  }

  async function showSweepChart(sweep: Sweep, ticket: number): Promise<void> {
    try {
      const published = await Promise.all(
        sweep.stops.map(async (stop, index) => trackForStop(index, stop)),
      );
      if (ticket !== state.openTicket) {
        return;
      }
      chartColumns = sweep.stops.map((stop, index) =>
        sweepColumn(stop, published[index] ?? [], sweep.landfall_step ?? null),
      );
      drawSweepChart(sweepChart, sweepChartLayout(chartColumns));
      const selected = sweep.stops.find((stop) => stop.id === sweepInput.dataset["stop"]);
      if (selected !== undefined) {
        markChartStop(selected);
      }
    } catch (error) {
      if (ticket !== state.openTicket) {
        return;
      }
      sweepChartReadout.textContent = `The displacement chart could not be drawn: ${describe(error)}`;
    }
    sweepChartSection.hidden = false;
  }

  async function selectStop(index: number, fromPointer: boolean): Promise<void> {
    const sweep = state.sweep;
    if (sweep === null) {
      return;
    }
    const bounded = Math.min(Math.max(Math.round(index), 0), sweep.stops.length - 1);
    const stop = sweep.stops[bounded];
    if (stop === undefined) {
      return;
    }
    sweepInput.value = String(bounded);
    setSweepLabel(stop);
    const ticket = ++state.stopTicket;
    summary.textContent = `Loading the ${sweepStopLabel(stop)} tracks.`;
    try {
      const tracks = await trackForStop(bounded, stop);
      if (ticket !== state.stopTicket || state.globe === null) {
        return;
      }
      state.tracks = tracks;
      state.trackProblem = null;
      state.globe.setTracks(seriesFromTracks(tracks));
    } catch (error) {
      if (ticket === state.stopTicket) {
        summary.textContent = `The ${sweepStopLabel(stop)} tracks could not be loaded: ${describe(error)}`;
      }
      return;
    }
    state.divergence =
      stop.divergence === undefined ? [] : await loadStopDivergence(state.id, stop.divergence);
    if (ticket !== state.stopTicket) {
      return;
    }
    updateReadout();
    syncRoute();
    if (fromPointer) {
      requestAnimationFrame(() => {
        for (const neighbour of [bounded - 1, bounded + 1]) {
          const next = sweep.stops[neighbour];
          if (next !== undefined) {
            void trackForStop(neighbour, next).catch(() => undefined);
          }
        }
      });
    }
  }

  function configureSweep(sweep: Sweep | null): void {
    state.stopTracks.clear();
    if (sweep === null || sweep.stops.length === 0) {
      displacementField.hidden = true;
      delete sweepInput.dataset["stop"];
      sweepInput.max = "0";
      sweepOutput.value = "";
      return;
    }
    displacementField.hidden = false;
    sweepInput.min = "0";
    sweepInput.max = String(sweep.stops.length - 1);
    sweepInput.step = "1";
    sweepInput.value = "0";
    const first = sweep.stops[0];
    if (first !== undefined) {
      setSweepLabel(first);
    }
  }
  return {
    selectStop,
    configureSweep,
    showSweepChart,
    close: () => {
      chartColumns = [];
      sweepChart.replaceChildren();
      sweepChartReadout.textContent = "";
      sweepChartSection.hidden = true;
    },
  };
}
