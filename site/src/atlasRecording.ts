import type { GlobeRenderer } from "./globeRenderer";
import type { AtlasState } from "./atlasState";
import { maxStep } from "./atlasState";
import { focusBasin } from "./atlasCamera";
import { recordFrame } from "./record";
import { element } from "./dom";

export function createAtlasRecording(
  state: AtlasState,
  panel: HTMLElement,
  setFrame: (frame: number, sync: boolean) => void,
  selectStop: (index: number, fromPointer: boolean) => Promise<void>,
) {
  const content = element("#record-content", HTMLTemplateElement);
  let recordActive = false;
  let recordCleanup: (() => void) | null = null;
  function exitRecordMode(): void {
    state.record = false;
    if (!recordActive) {
      return;
    }
    recordActive = false;
    document.documentElement.removeAttribute("data-record");
    recordCleanup?.();
    recordCleanup = null;
  }

  function startRecordMode(target: GlobeRenderer): void {
    if (recordActive) {
      return;
    }
    document.documentElement.dataset["record"] = "";
    recordActive = true;
    const previousPosition = panel.style.position;
    panel.style.position = "relative";
    const overlay = document.createElement("div");
    overlay.id = "record-still";
    overlay.append(content.content.cloneNode(true));
    panel.append(overlay);
    focusBasin(target, panel, state.tracks);
    target.controls.enabled = false;
    const sweep = state.sweep;
    const stopCount = sweep === null ? 0 : sweep.stops.length;
    let lastStop = 0;
    let lastStep = 0;
    const started = performance.now();
    const tick = (now: number): void => {
      const frame = recordFrame(now - started, stopCount, maxStep(state));
      if (frame.stop !== lastStop) {
        lastStop = frame.stop;
        if (sweep !== null) {
          void selectStop(frame.stop, false);
        }
      }
      if (frame.step !== lastStep) {
        lastStep = frame.step;
        setFrame(frame.step, false);
      }
      overlay.style.opacity = String(frame.stillOpacity);
      if (!frame.done) {
        requestAnimationFrame(tick);
      }
    };
    requestAnimationFrame(tick);
    recordCleanup = () => {
      panel.style.position = previousPosition;
      overlay.remove();
      target.controls.enabled = true;
    };
  }
  return { start: startRecordMode, close: exitRecordMode };
}
