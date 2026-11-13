/**
 * Colours used by the globe and the colour ramps.
 *
 * Ramps keep a monotone lightness order, so they stay readable under the common forms of
 * colour blindness.
 */
export type Rgb = readonly [number, number, number];
export type Ramp = "scalar" | "speed" | "diverging";

export const BACKGROUND_COLOUR = "#061719";
export const SEA_COLOUR = "#0a2327";
export const COASTLINE_COLOUR = "#9fc9c5";
export const OUTLINE_COLOUR = "#4f7b79";
export const CONTOUR_COLOUR = "#d6f3f4";

const TRACK_COLOURS: Readonly<Record<string, string>> = {
  baseline: "#d6f3f4",
  fork: "#ff5148",
  control: "#86b3a5",
};
const OTHER_TRACK_COLOUR = "#d0d0d0";

export function trackColour(run: string): string {
  return TRACK_COLOURS[run] ?? OTHER_TRACK_COLOUR;
}

/** Deep water to ice: lightness rises with the value, so it reads on the dark globe. */
const SCALAR_STOPS: readonly Rgb[] = [
  [11, 43, 51],
  [22, 86, 95],
  [40, 128, 132],
  [95, 179, 176],
  [200, 238, 232],
];

/** Calm air stays dark and the strongest wind is brightest. */
const SPEED_STOPS: readonly Rgb[] = SCALAR_STOPS;

/** Ice blue to the ground tone to signal red, with the dark centre reserved for no change. */
const DIVERGING_STOPS: readonly Rgb[] = [
  [95, 195, 232],
  [22, 44, 48],
  [255, 81, 72],
];

const RAMPS: Readonly<Record<Ramp, readonly Rgb[]>> = {
  scalar: SCALAR_STOPS,
  speed: SPEED_STOPS,
  diverging: DIVERGING_STOPS,
};

export const LUT_SIZE = 256;

function clamp01(value: number): number {
  return Math.min(Math.max(value, 0), 1);
}

function mixChannel(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

/** Colour at a normalised ramp position, with linear interpolation between stops. */
export function rampColour(ramp: Ramp, t: number): Rgb {
  const stops = RAMPS[ramp];
  const position = clamp01(t) * (stops.length - 1);
  const index = Math.min(Math.floor(position), stops.length - 2);
  const lower = stops[index];
  const upper = stops[index + 1];
  if (lower === undefined || upper === undefined) {
    throw new Error(`${ramp} ramp has no stops`);
  }
  const local = position - index;
  return [
    mixChannel(lower[0], upper[0], local),
    mixChannel(lower[1], upper[1], local),
    mixChannel(lower[2], upper[2], local),
  ];
}

/** Entry selected for a normalised value. The shader uses the same mapping. */
export function lutIndex(t: number): number {
  return Math.round(clamp01(t) * (LUT_SIZE - 1));
}

/** RGBA bytes for a one-row LUT texture. */
export function rampLut(ramp: Ramp): Uint8Array {
  const data = new Uint8Array(LUT_SIZE * 4);
  for (let index = 0; index < LUT_SIZE; index += 1) {
    const [red, green, blue] = rampColour(ramp, index / (LUT_SIZE - 1));
    data[index * 4] = red;
    data[index * 4 + 1] = green;
    data[index * 4 + 2] = blue;
    data[index * 4 + 3] = 255;
  }
  return data;
}
