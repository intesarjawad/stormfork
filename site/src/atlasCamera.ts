import type { GlobeRenderer } from "./globeRenderer";
import type { AtlasState } from "./atlasState";
import { lonLatToPosition, type TrackSample } from "./tracks";

const OPENING_MS = 2600;
const OPENING_START = 2.4;

export function focusPoint(
  target: GlobeRenderer,
  panel: HTMLElement,
  lat: number,
  lon: number,
): void {
  const [x, y, z] = lonLatToPosition(lat, lon);
  // On a portrait panel the horizontal field is narrow, so back off until the globe fits.
  const aspect = panel.clientWidth / Math.max(1, panel.clientHeight);
  const fit = aspect < 1 ? 1 + 1.15 / Math.sin(Math.atan(Math.tan(Math.PI / 8) * aspect)) : 0;
  const distance = Math.min(Math.max(target.camera.position.length(), fit), 7);
  target.camera.position.set(x * distance, y * distance, z * distance);
  target.controls.target.set(x, y, z);
  target.controls.update();
}

export function openingMove(
  target: GlobeRenderer,
  surface: HTMLCanvasElement,
  state: AtlasState,
  reducedMotion: boolean,
): void {
  if (reducedMotion) {
    return;
  }
  const end = target.camera.position.length();
  const direction = target.camera.position.clone().normalize();
  const began = performance.now();
  let cancelled = false;
  surface.addEventListener(
    "pointerdown",
    () => {
      cancelled = true;
    },
    { once: true },
  );
  const tick = (now: number): void => {
    if (cancelled || target !== state.globe) {
      return;
    }
    const t = Math.min(1, (now - began) / OPENING_MS);
    const eased = 1 - (1 - t) ** 3;
    target.camera.position
      .copy(direction)
      .multiplyScalar(OPENING_START + (end - OPENING_START) * eased);
    target.controls.update();
    target.requestRender();
    if (t < 1) {
      requestAnimationFrame(tick);
    }
  };
  requestAnimationFrame(tick);
}

export function focusBasin(
  target: GlobeRenderer,
  panel: HTMLElement,
  tracks: readonly TrackSample[],
): void {
  const sample = tracks[0];
  if (sample === undefined) {
    return;
  }
  focusPoint(target, panel, sample.lat, sample.lon);
}

export function markFirstFrame(): void {
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      performance.mark("stormfork:first-frame");
    });
  });
}
