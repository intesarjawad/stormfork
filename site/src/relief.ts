import {
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  CustomBlending,
  DataTexture,
  HalfFloatType,
  LinearFilter,
  MaxEquation,
  Mesh,
  NearestFilter,
  OneFactor,
  OrthographicCamera,
  PerspectiveCamera,
  PlaneGeometry,
  Points,
  RedFormat,
  RepeatWrapping,
  Scene,
  ShaderMaterial,
  UnsignedByteType,
  Vector2,
  Vector3,
  WebGLRenderTarget,
  WebGLRenderer,
  type IUniform,
  type Texture,
} from "three";

import { ATLANTIC_COAST } from "./atlanticCoast";
import { positionAt, type Storm } from "./fates";
import type { DecodedAtlas } from "./pressure";

/**
 * The relief: one run's real sea-level pressure as a 1-bit landscape, with every member's
 * track burnt into it.
 *
 * Five passes at the canvas's own low resolution, so each texel is one visible pixel:
 * the displaced surface (shade and line flags), the tracks, the heat field (ping-pong,
 * fed by stamps where the storms move), the ordered-dither composite, and a blit that
 * can dissolve from the previous frame through the same Bayer matrix.
 */

export type ReliefRun = "baseline" | "fork";

export interface ReliefInput {
  readonly canvas: HTMLCanvasElement;
  readonly atlases: Readonly<Record<ReliefRun, DecodedAtlas>>;
  /** Pressure range in Pa that the atlas bytes span. */
  readonly range: readonly [number, number];
  readonly storms: readonly Storm[];
  readonly lastStep: number;
  /** The page's two inks, as hex. */
  readonly inks: readonly [string, string];
}

export interface ReliefState {
  readonly step: number;
  readonly run: ReliefRun;
  /** The storm whose track burns alone, or null for every storm of the runs shown. */
  readonly focus: Storm | null;
}

/** Each relief pixel is this many CSS pixels wide, on every screen, so the dither has one pitch. */
const PIXEL = 2;

// The surface runs past every edge of the frame, so no edge of the world ever shows.
const WEST = -114;
const EAST = 4;
const SOUTH = 4;
const NORTH = 78;
/** What the frame is aimed at: Sandy from the Bahamas to the Great Lakes, and the moved storms well out to sea. */
const VIEW = { lon: -66, lat: 35, west: -100, east: -32 } as const;
/** The storms the relief keeps its contrast around; the rest of the basin flattens. */
const NEAR_MAX = 16;
const LON_C = (WEST + EAST) / 2;
const LAT_C = (SOUTH + NORTH) / 2;
const KX = Math.cos((LAT_C * Math.PI) / 180);
const TILE_W = 361;
const TILE_H = 181;
const COLUMNS = 5;
/** Relief height per hPa: highs swell gently, lows sink as pits. */
const HIGH_LIFT = 0.06;
const LOW_DEPTH = 0.2;
const SEGMENT_SAMPLES = 28;
/** The ribbon's width at its hot end and as a cold scar, in relief pixels (2 CSS px each). */
const RIBBON_PX = 5;
const SCAR_PX = 3;
/** Where a ribbon's bands end, in six-hour steps since the fire passed. */
const HEAD_STEPS = 0.2;
const BAND_STEPS = 1;
const EMBER_STEPS = 6;
const SCAR_STEPS = 16;

/**
 * A ribbon's heat where the fire passed `age` steps ago, read by the composite's bands: above
 * 0.92 white-hot, above 0.66 the inverted band, above 0.3 embers, above 0.06 char.
 */
export function heatAt(age: number): number {
  if (age < HEAD_STEPS) return 1;
  if (age < BAND_STEPS) return 0.8;
  if (age < EMBER_STEPS) return 0.62 - 0.32 * ((age - BAND_STEPS) / (EMBER_STEPS - BAND_STEPS));
  return 0.26 - 0.14 * Math.min((age - EMBER_STEPS) / (SCAR_STEPS - EMBER_STEPS), 1);
}
const MAX_POINTS = 48_000;
const BAYER_8 = [
  0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26, 12, 44, 4, 36, 14, 46, 6, 38, 60, 28,
  52, 20, 62, 30, 54, 22, 3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25, 15, 47, 7,
  39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21,
];

const SURFACE = /* glsl */ `
  uniform sampler2D uAtlas;
  uniform sampler2D uLand;
  uniform vec2 uAtlasSize;
  uniform vec2 uRange;
  uniform float uStep;

  const float LON_C = ${LON_C.toFixed(1)};
  const float LAT_C = ${LAT_C.toFixed(1)};
  const float KX = ${KX.toFixed(6)};

  vec2 lonLatOf(vec3 p) { return vec2(LON_C + p.x / KX, LAT_C - p.z); }

  float tileValue(float frame, vec2 ll) {
    vec2 tile = vec2(mod(frame, ${COLUMNS.toFixed(1)}) * ${TILE_W.toFixed(1)},
      floor(frame / ${COLUMNS.toFixed(1)}) * ${TILE_H.toFixed(1)});
    vec2 cell = vec2(mod(ll.x + 360.0, 360.0), ll.y + 90.0);
    return texture2D(uAtlas, (tile + cell + 0.5) / uAtlasSize).r;
  }

  // Sea-level pressure in hPa, interpolated between the two six-hour frames around uStep.
  float hpaAt(vec2 ll) {
    float f = floor(uStep);
    float v = mix(tileValue(f, ll), tileValue(min(f + 1.0, ${String(40)}.0), ll), uStep - f);
    return (uRange.x + v * (uRange.y - uRange.x)) / 100.0;
  }

  float landAt(vec2 ll) {
    return texture2D(uLand, vec2((ll.x - ${WEST.toFixed(1)}) / ${(EAST - WEST).toFixed(1)},
      (ll.y - ${SOUTH.toFixed(1)}) / ${(NORTH - SOUTH).toFixed(1)})).r;
  }

  // How close a point is to any storm on show, 1 at a storm and 0 well clear of all of them.
  uniform vec2 uNear[${String(NEAR_MAX)}];
  uniform float uNearCount;
  float nearness(vec2 ll) {
    float w = 0.0;
    for (int i = 0; i < ${String(NEAR_MAX)}; i++) {
      if (float(i) >= uNearCount) break;
      w = max(w, 1.0 - smoothstep(6.0, 15.0, length((ll - uNear[i]) * vec2(KX, 1.0))));
    }
    return w;
  }

  // Lows sink as pits, highs swell gently, land stands a step above the sea.
  float heightAt(float hpa, float land) {
    float d = hpa - 1012.0;
    return (d > 0.0 ? d * ${HIGH_LIFT.toFixed(3)} : d * ${LOW_DEPTH.toFixed(3)}) + land * 0.6;
  }
`;

const RELIEF_VERTEX = /* glsl */ `
  ${SURFACE}
  varying vec2 vLL;
  void main() {
    vec3 p = position;
    vec2 ll = lonLatOf(p);
    p.y = heightAt(hpaAt(ll), landAt(ll));
    vLL = ll;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }
`;

const RELIEF_FRAGMENT = /* glsl */ `
  ${SURFACE}
  varying vec2 vLL;
  void main() {
    float land = landAt(vLL);
    float isLand = step(0.5, land);
    // Normal from the field a degree either side, not from the mesh, so the shading follows
    // the weather's broad shapes rather than the grid's.
    vec2 e = vec2(1.2, 0.0);
    float hE = heightAt(hpaAt(vLL + e), landAt(vLL + e));
    float hW = heightAt(hpaAt(vLL - e), landAt(vLL - e));
    float hN = heightAt(hpaAt(vLL + e.yx), landAt(vLL + e.yx));
    float hS = heightAt(hpaAt(vLL - e.yx), landAt(vLL - e.yx));
    vec3 n = normalize(vec3(-(hE - hW) / (2.0 * e.x * KX), 1.0, (hN - hS) / (2.0 * e.x)));
    // Light from the north-west, low, so a pit's far wall catches it and its near wall falls dark.
    vec3 sun = normalize(vec3(-0.5, 0.55, -0.67));
    float light = dot(n, sun) - sun.y;
    // Sea pale over the highs and dark in the lows, land a level above it.
    float hpa = (hpaAt(vLL) * 2.0 + hpaAt(vLL + e) + hpaAt(vLL - e) + hpaAt(vLL + e.yx)
      + hpaAt(vLL - e.yx)) / 6.0;
    // Away from the storms the light and land's step above the sea are halved, not removed:
    // land, sea and the north-west light keep their order, and only the storms keep full contrast.
    float near = nearness(vLL);
    float sea = mix(0.06, 0.17, smoothstep(990.0, 1028.0, hpa));
    float base = mix(sea, mix(0.5 * (sea + 0.25), 0.25, near), isLand);
    float shade = base + light * 1.9;
    shade = mix(base, shade, mix(0.5, 1.0, near));

    // Isobars every 4 hPa below 1000 hPa, over the sea and near a storm: the storms, ringed.
    float c = hpa / 4.0;
    // Every fifth ring (each 20 hPa) is heavier and solid, the rest fine and dotted, so the
    // rings read as a pressure field rather than a target.
    float major = 1.0 - step(0.5, abs(mod(floor(c + 0.5), 5.0)));
    float onContour = 1.0 - step(mix(0.6, 1.3, major), abs(fract(c - 0.5) - 0.5) / max(fwidth(c), 1e-4));
    float onCoast = 1.0 - step(0.9, abs(land - 0.5) / max(fwidth(land), 1e-4));
    // Rings go solid near a storm and dotted at the edge of its reach, so none ends abruptly.
    float ringed = onContour * (1.0 - step(1000.0, hpa)) * (1.0 - isLand)
      * (step(0.6, near) * 0.5 + step(0.3, near) * 0.5) * mix(0.5, 1.0, major);
    // Line flag: 1 the coast, a solid line that every frame keeps; 0.8 a solid ring, 0.4 a
    // dotted one.
    float lines = max(ringed * 0.8, onCoast);
    // Blue marks the surface: 1 the sea, 0.75 the land, which is dithered in its own texture.
    gl_FragColor = vec4(clamp(shade, 0.0, 1.0), lines, mix(1.0, 0.75, isLand), 1.0);
  }
`;

// Marks and heat stamps share one vertex shader: a point on the surface at (lon, lat).
const POINT_VERTEX = /* glsl */ `
  ${SURFACE}
  attribute vec2 aLL;
  attribute vec4 aValue;
  attribute float aSize;
  varying vec4 vValue;
  void main() {
    vec3 p = vec3((aLL.x - LON_C) * KX, 0.0, LAT_C - aLL.y);
    p.y = heightAt(hpaAt(aLL), landAt(aLL)) + 0.08;
    vValue = aValue;
    gl_PointSize = aSize;
    gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
  }
`;

// b: a storm's head, drawn over the fire; a: its knock-out.
const MARK_FRAGMENT = /* glsl */ `
  varying vec4 vValue;
  void main() { gl_FragColor = vValue; }
`;

const STAMP_FRAGMENT = /* glsl */ `
  uniform float uTime;
  varying vec4 vValue;
  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  void main() {
    // Square, pixel-edged stamps, hottest at the middle: a solid core, then a band that
    // inverts, then char. The rim is ragged and loses whole 2x2 blocks at random each tick.
    vec2 q = abs(gl_PointCoord - 0.5) * 2.0;
    float d = max(q.x, q.y) * (0.8 + 0.35 * hash(floor(gl_FragCoord.xy)));
    // A ribbon's stamps (green set) keep their rim, so the ribbon stays one even width.
    float rim = step(0.55, d) * (1.0 - step(0.3, hash(floor(gl_FragCoord.xy / 2.0) + floor(uTime * 15.0))))
      * (1.0 - vValue.g);
    // A hot stamp is hottest at its middle; a char stamp is coldest there, its edge held at
    // ember heat, so a scar reads as a dark furrow with a smouldering rim on any ground. A
    // track burning alone (blue set) crosses a dimmed relief, so its rim smoulders harder.
    float heat = vValue.r < 0.3 && d > 0.6 ? max(min(3.5 * vValue.r, 0.6), 0.5 * vValue.b) : vValue.r * (1.0 - 0.35 * d);
    float t = heat * (1.0 - rim) * (1.0 - step(1.0, d));
    gl_FragColor = vec4(t, 0.0, 0.0, 1.0);
  }
`;

const QUAD_VERTEX = /* glsl */ `
  void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

const HEAT_FRAGMENT = /* glsl */ `
  uniform sampler2D uPrevious;
  uniform vec2 uRes;
  uniform float uTime;
  uniform float uDecay;
  uniform float uCool;
  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  void main() {
    vec2 px = gl_FragCoord.xy;
    // Heat climbs the screen in 3-pixel columns that flicker on their own clocks.
    float n = hash(floor(px / 3.0) + floor(uTime * 11.0) * vec2(1.3, 2.9));
    vec2 from = (px - vec2((n - 0.5) * 0.8, n * 0.35)) / uRes;
    float h = texture2D(uPrevious, from).r;
    gl_FragColor = vec4(max(h * uDecay - uCool, 0.0), 0.0, 0.0, 1.0);
  }
`;

const COMPOSITE_FRAGMENT = /* glsl */ `
  uniform sampler2D uRelief;
  uniform sampler2D uMarks;
  uniform sampler2D uHeat;
  uniform sampler2D uBayer;
  uniform vec2 uRes;
  uniform float uTime;
  uniform float uDim;
  uniform vec3 uInkA;
  uniform vec3 uInkB;
  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float bayer(vec2 px) { return texture2D(uBayer, (mod(floor(px), 8.0) + 0.5) / 8.0).r; }
  // Interleaved gradient noise spreads dots like blue noise: sea in it never forms the
  // screen-door grid the 8x8 matrix makes, and it reads apart from the hatched land.
  float ign(vec2 px) { return fract(52.9829189 * fract(dot(floor(px), vec2(0.06711056, 0.00583715)))); }
  void main() {
    vec2 px = gl_FragCoord.xy;
    vec2 uv = px / uRes;
    float heat = texture2D(uHeat, uv).r;
    float n4 = hash(floor(px / 4.0) + floor(uTime * 9.0) * vec2(1.7, 3.1));
    float n2 = hash(floor(px / 2.0) + floor(uTime * 14.0) * vec2(5.3, 1.1));
    float h = heat * (0.85 + 0.2 * n4 + 0.1 * n2);

    // The air above a burn shimmers: the relief is read a few pixels off.
    vec2 shimmer = vec2((n2 - 0.5) * 2.0, (n4 - 0.5) * 3.0) * min(heat, 1.0) / uRes;
    vec4 r = texture2D(uRelief, uv + shimmer);
    // While one storm burns alone the rest of the relief sinks to half its tone and
    // its rings thin to dots; the coast stays whole.
    float shade = r.b > 0.5 ? r.r * mix(1.0, 0.5, uDim) : 0.0;
    // Land is set in a coarser grain than the sea: the same noise read on 2x2 cells, so the
    // two read apart by texture without stripes.
    float t = ign(r.b > 0.5 && r.b < 0.9 ? floor(px / 2.0) + 0.5 : px);
    float surface = step(0.5, r.b);
    float ink = surface * step(t, max(shade, 0.03));
    float lines = r.b > 0.5 ? (r.g > 0.9 ? 1.0 : r.g * mix(1.0, 0.5, uDim)) : 0.0;
    bool onLine = lines > 0.75 || (lines > 0.3 && mod(px.x + px.y, 2.0) < 1.0);
    if (onLine) ink = 1.0;

    vec4 m = texture2D(uMarks, uv);

    // Burning, in two inks, banded by heat (see heatAt): white-hot; the band where the fire
    // eats the image, lifted to a half tone first so it reads on dark sea; embers in 4 px
    // blocks thinning as they cool; then char, a knock-out to the ground with sparse 2 px
    // embers, darker than any ground around it, so it reads as burnt and never as snow.
    if (h > 0.92) ink = 1.0;
    else if (h > 0.66) ink = 1.0 - step(bayer(px), clamp(r.r + 0.35, 0.0, 1.0));
    else if (h > 0.3) {
      float embers = mix(0.2, 0.6, (h - 0.3) / 0.36);
      ink = step(1.0 - embers, hash(floor(px / 2.0) + floor(uTime * 20.0)));
    } else if (h > 0.06) {
      ink = step(1.0 - mix(0.0, 0.06, (h - 0.06) / 0.24), hash(floor(px) + floor(uTime * 6.0)));
    }
    // The coast is drawn over the fire, so a burn never erases the land it crosses; the heads
    // are drawn over both, each in its own knock-out.
    if (r.b > 0.5 && r.g > 0.9) ink = 1.0;
    if (m.a > 0.5) ink = 0.0;
    if (m.b > 0.5) ink = 1.0;

    // Alpha marks the relief's lines, so a dissolve can keep both runs' isobars whole.
    gl_FragColor = vec4(mix(uInkA, uInkB, ink), onLine ? 0.5 : 1.0);
  }
`;

const BLIT_FRAGMENT = /* glsl */ `
  uniform sampler2D uNow;
  uniform sampler2D uBefore;
  uniform sampler2D uBayer;
  uniform vec2 uRes;
  uniform float uDissolve;
  uniform vec3 uInk;
  void main() {
    vec2 px = gl_FragCoord.xy;
    vec2 uv = px / uRes;
    // The next instant arrives through the dither's own matrix, threshold by threshold, so
    // the two images mix as tones and never beat against each other.
    float t = texture2D(uBayer, (mod(floor(px), 8.0) + 0.5) / 8.0).r;
    vec4 now = texture2D(uNow, uv);
    vec4 before = texture2D(uBefore, uv);
    vec3 color = t < uDissolve ? now.rgb : before.rgb;
    if (uDissolve < 1.0 && min(now.a, before.a) < 0.75) color = uInk;
    gl_FragColor = vec4(color, 1.0);
  }
`;

function rgb(hex: string): Vector3 {
  const value = Number.parseInt(hex.trim().replace("#", ""), 16);
  return new Vector3(((value >> 16) & 255) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255);
}

function landTexture(): CanvasTexture {
  const scale = 10;
  const canvas = document.createElement("canvas");
  canvas.width = (EAST - WEST) * scale;
  canvas.height = (NORTH - SOUTH) * scale;
  const context = canvas.getContext("2d");
  if (context === null) {
    throw new Error("a 2D canvas is not available to draw the land mask");
  }
  context.fillStyle = "#000";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.beginPath();
  for (const ring of ATLANTIC_COAST) {
    for (let i = 0; i < ring.length; i += 2) {
      const x = ((ring[i] ?? 0) - WEST) * scale;
      const y = (NORTH - (ring[i + 1] ?? 0)) * scale;
      if (i === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.closePath();
  }
  context.fillStyle = "#fff";
  context.fill("evenodd");
  const texture = new CanvasTexture(canvas);
  texture.minFilter = LinearFilter;
  texture.magFilter = LinearFilter;
  return texture;
}

function atlasTexture(atlas: DecodedAtlas): DataTexture {
  const texture = new DataTexture(
    atlas.bytes,
    atlas.width,
    atlas.height,
    RedFormat,
    UnsignedByteType,
  );
  texture.unpackAlignment = 1;
  texture.minFilter = LinearFilter;
  texture.magFilter = LinearFilter;
  texture.needsUpdate = true;
  return texture;
}

function bayerTexture(): DataTexture {
  const bytes = new Uint8Array(BAYER_8.map((v) => Math.round(((v + 0.5) / 64) * 255)));
  const texture = new DataTexture(bytes, 8, 8, RedFormat, UnsignedByteType);
  texture.unpackAlignment = 1;
  texture.minFilter = NearestFilter;
  texture.magFilter = NearestFilter;
  texture.wrapS = RepeatWrapping;
  texture.wrapT = RepeatWrapping;
  texture.needsUpdate = true;
  return texture;
}

function target(halfFloat = false): WebGLRenderTarget {
  return new WebGLRenderTarget(1, 1, {
    minFilter: NearestFilter,
    magFilter: NearestFilter,
    depthBuffer: !halfFloat,
    ...(halfFloat ? { type: HalfFloatType } : {}),
  });
}

/** A growable list of surface points, drawn as GL points. */
class PointBatch {
  readonly geometry = new BufferGeometry();
  private readonly ll = new Float32Array(MAX_POINTS * 2);
  private readonly value = new Float32Array(MAX_POINTS * 4);
  private readonly size = new Float32Array(MAX_POINTS);
  private count = 0;

  constructor() {
    this.geometry.setAttribute("aLL", new BufferAttribute(this.ll, 2));
    this.geometry.setAttribute("aValue", new BufferAttribute(this.value, 4));
    this.geometry.setAttribute("aSize", new BufferAttribute(this.size, 1));
    // The vertex shader places every point itself; a position attribute only sets the count.
    this.geometry.setAttribute(
      "position",
      new BufferAttribute(new Float32Array(MAX_POINTS * 3), 3),
    );
  }

  reset(): void {
    this.count = 0;
  }

  add(lon: number, lat: number, size: number, r: number, g = 0, b = 0, a = 0): void {
    if (this.count >= MAX_POINTS) return;
    const i = this.count++;
    this.ll[i * 2] = lon;
    this.ll[i * 2 + 1] = lat;
    this.value.set([r, g, b, a], i * 4);
    this.size[i] = size;
  }

  commit(): void {
    for (const name of ["aLL", "aValue", "aSize"]) {
      const attribute = this.geometry.getAttribute(name) as BufferAttribute;
      attribute.needsUpdate = true;
    }
    this.geometry.setDrawRange(0, this.count);
  }
}

export class Relief {
  private readonly renderer: WebGLRenderer;
  private readonly camera = new PerspectiveCamera(30, 1, 1, 400);
  private readonly flat = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly surfaceUniforms: Record<string, IUniform>;
  private readonly textures: Texture[] = [];
  private readonly atlases: Readonly<Record<ReliefRun, DataTexture>>;
  private readonly bytes: Readonly<Record<ReliefRun, DecodedAtlas>>;
  private readonly reliefTarget = target();
  private readonly marksTarget = target();
  private readonly heatTargets = [target(true), target(true)] as const;
  private readonly outTargets = [target(), target()] as const;
  private readonly reliefScene = new Scene();
  private readonly marksScene = new Scene();
  private readonly stampScene = new Scene();
  private readonly heatScene = new Scene();
  private readonly compositeScene = new Scene();
  private readonly blitScene = new Scene();
  private readonly marks = new PointBatch();
  private readonly stamps = new PointBatch();
  private readonly heatUniforms: Record<string, IUniform>;
  private readonly compositeUniforms: Record<string, IUniform>;
  private readonly blitUniforms: Record<string, IUniform>;
  private readonly stampUniforms: Record<string, IUniform>;
  private heatIndex = 0;
  private outIndex = 0;
  private width = 1;
  private height = 1;
  private cssWidth = 1;
  private cssHeight = 1;
  private state: ReliefState;
  private lastStep: number;
  private dissolveStart = -1;
  private burnStarted = 0;
  private readonly range: readonly [number, number];
  private readonly storms: readonly Storm[];

  constructor(input: ReliefInput) {
    this.renderer = new WebGLRenderer({
      canvas: input.canvas,
      antialias: false,
      alpha: false,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(1);
    this.renderer.autoClear = true;
    // Every target starts empty: no relief (sky), no marks, no heat.
    this.renderer.setClearColor(0x000000, 0);
    this.range = input.range;
    this.storms = input.storms;
    this.lastStep = input.lastStep;
    this.bytes = input.atlases;
    this.atlases = {
      baseline: atlasTexture(input.atlases.baseline),
      fork: atlasTexture(input.atlases.fork),
    };
    const land = landTexture();
    const bayer = bayerTexture();
    this.textures.push(this.atlases.baseline, this.atlases.fork, land, bayer);
    this.state = { step: 0, run: "baseline", focus: null };

    this.surfaceUniforms = {
      uAtlas: { value: this.atlases.baseline },
      uLand: { value: land },
      uAtlasSize: { value: [input.atlases.baseline.width, input.atlases.baseline.height] },
      uRange: { value: [input.range[0], input.range[1]] },
      uStep: { value: 0 },
      uNear: { value: Array.from({ length: NEAR_MAX }, () => new Vector2()) },
      uNearCount: { value: 0 },
    };
    // About one vertex per relief pixel where the storms are.
    const plane = new PlaneGeometry((EAST - WEST) * KX, NORTH - SOUTH, 480, 360);
    plane.rotateX(-Math.PI / 2);
    this.reliefScene.add(
      new Mesh(
        plane,
        new ShaderMaterial({
          uniforms: this.surfaceUniforms,
          vertexShader: RELIEF_VERTEX,
          fragmentShader: RELIEF_FRAGMENT,
        }),
      ),
    );

    const pointMaterial = (fragmentShader: string, extra: Record<string, IUniform>, add: boolean) =>
      new ShaderMaterial({
        uniforms: { ...this.surfaceUniforms, ...extra },
        vertexShader: POINT_VERTEX,
        fragmentShader,
        depthTest: false,
        depthWrite: false,
        transparent: add,
        // Stamps keep the hotter of what is there and what lands: overlapping storms never blow out.
        ...(add
          ? {
              blending: CustomBlending,
              blendEquation: MaxEquation,
              blendSrc: OneFactor,
              blendDst: OneFactor,
            }
          : {}),
      });
    this.marksScene.add(new Points(this.marks.geometry, pointMaterial(MARK_FRAGMENT, {}, false)));
    this.stampUniforms = { uTime: { value: 0 } };
    const stampPoints = new Points(
      this.stamps.geometry,
      pointMaterial(STAMP_FRAGMENT, this.stampUniforms, true),
    );
    stampPoints.frustumCulled = false;
    this.stampScene.add(stampPoints);
    for (const child of this.marksScene.children) child.frustumCulled = false;

    const quad = (uniforms: Record<string, IUniform>, fragmentShader: string): Mesh =>
      new Mesh(
        new PlaneGeometry(2, 2),
        new ShaderMaterial({
          uniforms,
          vertexShader: QUAD_VERTEX,
          fragmentShader,
          depthTest: false,
          depthWrite: false,
        }),
      );
    this.heatUniforms = {
      uPrevious: { value: null },
      uRes: { value: [1, 1] },
      uTime: { value: 0 },
      uDecay: { value: 0.96 },
      uCool: { value: 0.002 },
    };
    this.heatScene.add(quad(this.heatUniforms, HEAT_FRAGMENT));
    const [inkA, inkB] = input.inks;
    this.compositeUniforms = {
      uRelief: { value: this.reliefTarget.texture },
      uMarks: { value: this.marksTarget.texture },
      uHeat: { value: this.heatTargets[0].texture },
      uBayer: { value: bayer },
      uRes: { value: [1, 1] },
      uTime: { value: 0 },
      uDim: { value: 0 },
      uInkA: { value: rgb(inkA) },
      uInkB: { value: rgb(inkB) },
    };
    this.compositeScene.add(quad(this.compositeUniforms, COMPOSITE_FRAGMENT));
    this.blitUniforms = {
      uNow: { value: this.outTargets[0].texture },
      uBefore: { value: this.outTargets[1].texture },
      uBayer: { value: bayer },
      uRes: { value: [1, 1] },
      uDissolve: { value: 1 },
      uInk: { value: rgb(inkB) },
    };
    this.blitScene.add(quad(this.blitUniforms, BLIT_FRAGMENT));
  }

  /**
   * Fit the relief to its box at its pixel size. The canvas is drawn a whole number of
   * relief pixels across, up to one past the box's edge, so every relief pixel is exactly
   * PIXEL CSS pixels and the dither never changes pitch.
   */
  resize(cssWidth: number, cssHeight: number): void {
    this.width = Math.max(1, Math.ceil(cssWidth / PIXEL));
    this.height = Math.max(1, Math.ceil(cssHeight / PIXEL));
    this.cssWidth = this.width * PIXEL;
    this.cssHeight = this.height * PIXEL;
    this.renderer.setSize(this.width, this.height, false);
    const canvas = this.renderer.domElement;
    canvas.style.width = `${String(this.cssWidth)}px`;
    canvas.style.height = `${String(this.cssHeight)}px`;
    for (const t of [
      this.reliefTarget,
      this.marksTarget,
      ...this.heatTargets,
      ...this.outTargets,
    ]) {
      t.setSize(this.width, this.height);
    }
    for (const uniforms of [this.heatUniforms, this.compositeUniforms, this.blitUniforms]) {
      const res = uniforms["uRes"];
      if (res) res.value = [this.width, this.height];
    }
    this.frameCamera();
    this.clearHeat();
  }

  /** Aim the camera so the view's longitudes fill the width, looking down from the south. */
  private frameCamera(): void {
    const aspect = this.width / this.height;
    const hfov = 50;
    const pitch = ((aspect < 1 ? 50 : 42) * Math.PI) / 180;
    // A phone frames a narrower stretch, so its storms stay big enough to find unlabelled.
    const half = ((VIEW.east - VIEW.west) / 2) * KX * (aspect < 1 ? 0.78 : 1);
    const distance = half / Math.tan((hfov * Math.PI) / 360);
    const x = (VIEW.lon - LON_C) * KX;
    const z = LAT_C - VIEW.lat;
    this.camera.aspect = aspect;
    this.camera.fov = (2 * Math.atan(Math.tan((hfov * Math.PI) / 360) / aspect) * 180) / Math.PI;
    this.camera.position.set(x, -2 + distance * Math.sin(pitch), z + distance * Math.cos(pitch));
    this.camera.lookAt(x, -2, z);
    this.camera.updateProjectionMatrix();
    this.camera.updateMatrixWorld();
  }

  private clearHeat(): void {
    for (const t of this.heatTargets) {
      this.renderer.setRenderTarget(t);
      this.renderer.clear();
    }
    this.renderer.setRenderTarget(null);
  }

  /** Begin a dither dissolve from what is on screen now to whatever is drawn next. */
  dissolve(now: number): void {
    this.outIndex = 1 - this.outIndex;
    this.dissolveStart = now;
  }

  set(next: ReliefState, now: number): void {
    const previous = this.state;
    if (next.focus !== previous.focus) {
      this.burnStarted = now;
    }
    this.state = next;
  }

  /**
   * Where a point on the surface lands on the page, in CSS pixels from the canvas corner; it
   * may lie past the canvas's edge. Null behind the camera.
   */
  project(lon: number, lat: number): { x: number; y: number } | null {
    const height = this.heightAt(lon, lat);
    const p = new Vector3((lon - LON_C) * KX, height + 0.08, LAT_C - lat).project(this.camera);
    if (p.z > 1) return null;
    return { x: ((p.x + 1) / 2) * this.cssWidth, y: ((1 - p.y) / 2) * this.cssHeight };
  }

  /**
   * How far from a storm its rings below 1000 hPa reach on screen, in CSS pixels, so a name's
   * leader can start outside them.
   */
  ringReach(lon: number, lat: number): number {
    const centre = this.project(lon, lat);
    if (centre === null) return 0;
    let reach = 0;
    for (let k = 0; k < 16; k += 1) {
      const a = (k * Math.PI) / 8;
      const along = (r: number): [number, number] => [
        lon + (Math.cos(a) * r) / KX,
        lat + Math.sin(a) * r,
      ];
      let r = 0;
      while (r < 15 && this.hpaAt(...along(r)) < 1000) r += 0.5;
      const edge = this.project(...along(r));
      if (edge !== null) reach = Math.max(reach, Math.hypot(edge.x - centre.x, edge.y - centre.y));
    }
    return reach;
  }

  /** Where the heavy rings (each 20 hPa) cross due south of a storm, to be labelled there. */
  ringLabels(lon: number, lat: number): { x: number; y: number; hpa: number }[] {
    const out: { x: number; y: number; hpa: number }[] = [];
    let previous = this.hpaAt(lon, lat);
    for (let r = 0.25; r < 15; r += 0.25) {
      const hpa = this.hpaAt(lon, lat - r);
      const ring = Math.ceil(previous / 20) * 20;
      if (ring <= hpa && ring < 1000) {
        const at = this.project(lon, lat - r);
        if (at !== null) out.push({ ...at, hpa: ring });
      }
      previous = hpa;
    }
    return out;
  }

  private hpaAt(lon: number, lat: number): number {
    const atlas = this.bytes[this.state.run];
    const cell = (frame: number, i: number, j: number): number =>
      atlas.bytes[
        (Math.floor(frame / COLUMNS) * TILE_H + j) * atlas.width + (frame % COLUMNS) * TILE_W + i
      ] ?? 0;
    // Bilinear between the four grid points around the position, as the shaders sample it.
    const value = (frame: number): number => {
      const x = ((lon % 360) + 360) % 360;
      const y = Math.min(Math.max(lat + 90, 0), TILE_H - 1);
      const i = Math.floor(x);
      const j = Math.floor(y);
      const fx = x - i;
      const fy = y - j;
      const row = (jj: number): number =>
        cell(frame, i, jj) * (1 - fx) + cell(frame, Math.min(i + 1, TILE_W - 1), jj) * fx;
      return row(j) * (1 - fy) + row(Math.min(j + 1, TILE_H - 1)) * fy;
    };
    const f = Math.floor(this.state.step);
    const byte = value(f) + (value(Math.min(f + 1, 40)) - value(f)) * (this.state.step - f);
    return (this.range[0] + (byte / 255) * (this.range[1] - this.range[0])) / 100;
  }

  private heightAt(lon: number, lat: number): number {
    const d = this.hpaAt(lon, lat) - 1012;
    return d > 0 ? d * HIGH_LIFT : d * LOW_DEPTH;
  }

  private visibleStorms(burning = true): { solid: Storm[]; twin: Storm[] } {
    const twin: ReliefRun = this.state.run === "fork" ? "baseline" : "fork";
    // A still frame and the closing frame keep every fate, the control's too.
    const closing = !burning || this.state.step >= this.lastStep;
    return {
      solid: this.storms.filter((s) => s.run === this.state.run),
      twin: this.storms.filter((s) => s.run === twin || (closing && s.run === "control")),
    };
  }

  private trace(
    storm: Storm,
    from: number,
    to: number,
    each: (lon: number, lat: number, at: number) => void,
  ) {
    if (to <= from) return;
    const samples = Math.max(2, Math.ceil((to - from) * SEGMENT_SAMPLES));
    for (let i = 0; i <= samples; i += 1) {
      const step = from + ((to - from) * i) / samples;
      const at = positionAt(storm, step);
      if (at !== null) each(at.lon, at.lat, step);
    }
  }

  /**
   * Where a storm's fire stands: at the storm, held at its last fix until the ledger strikes
   * the row (the step after it), then null.
   */
  private frontOf(storm: Storm, step: number): number | null {
    const last = storm.fixes.at(-1)?.step ?? 0;
    return Math.floor(step) > last ? null : Math.min(step, last);
  }

  /** How far the chosen track's fire has run, 2 s from its first fix to where the storm is. */
  private burnReach(now: number): number {
    const focus = this.state.focus;
    if (focus === null) return 0;
    const to = this.frontOf(focus, this.state.step) ?? focus.fixes.at(-1)?.step ?? 0;
    const progress = Math.min((now - this.burnStarted) / 2000, 1);
    return to * (1 - (1 - progress) ** 2);
  }

  /**
   * Stamp one storm's ribbon from the edit point to `end`, its heat set by how long ago the fire
   * passed each point at the instant `now`: an inverted band for the first step, embers that
   * thin to the sixth, then a thin dark scar. The ribbon narrows as it cools.
   */
  private fire(storm: Storm, end: number, now: number, scale = 1): void {
    this.trace(storm, 0, end, (lon, lat, at) => {
      const age = now - at;
      const cooling = Math.min(age / EMBER_STEPS, 1);
      const width = age < EMBER_STEPS ? RIBBON_PX - (RIBBON_PX - SCAR_PX) * cooling : SCAR_PX;
      this.stamps.add(lon, lat, width * scale, heatAt(age), 1, scale > 1 ? 1 : 0);
    });
  }

  private buildMarks(burning: boolean, now: number): void {
    const { step, focus } = this.state;
    const { solid, twin } = this.visibleStorms(burning);
    this.marks.reset();
    // Each track is its ribbon, in play and in a still alike: no lines, only the heads.
    if (focus === null) for (const storm of [...solid, ...twin]) this.head(storm, step);
    else this.head(focus, burning ? Math.min(this.burnReach(now), step) : step);
    this.marks.commit();
  }

  /** A live storm's head: a solid 8 px square in a knock-out, crisp over its own fire. */
  private head(storm: Storm, step: number): void {
    const front = this.frontOf(storm, step);
    const at = front === null ? null : positionAt(storm, front);
    if (at === null) return;
    this.marks.add(at.lon, at.lat, 6, 0, 0, 0, 1);
    this.marks.add(at.lon, at.lat, 4, 0, 0, 1, 0);
  }

  private buildStamps(now: number, burning: boolean): void {
    const { step, focus } = this.state;
    this.stamps.reset();
    if (focus === null) {
      const { solid, twin } = this.visibleStorms(burning);
      // A struck storm's ribbon ends where it was last seen and keeps cooling from there.
      for (const storm of [...solid, ...twin])
        this.fire(storm, this.frontOf(storm, step) ?? storm.fixes.at(-1)?.step ?? 0, step);
    } else {
      // The chosen track burns alone, half as large again, sweeping from the edit point to
      // where the storm is in 2 s, then cooling with the watch like the rest.
      const reach = burning ? this.burnReach(now) : Infinity;
      const end = this.frontOf(focus, step) ?? focus.fixes.at(-1)?.step ?? 0;
      this.fire(focus, Math.min(reach, end), Math.min(reach, step), 1.5);
    }
    this.stamps.commit();
  }

  /** Keep the relief's contrast around every storm on show that is still alive. */
  private placeNear(): void {
    const near = this.surfaceUniforms["uNear"]?.value as Vector2[] | undefined;
    const count = this.surfaceUniforms["uNearCount"];
    if (near === undefined || count === undefined) return;
    // Only the run whose pressure is drawn: rings near the other run's heads would be stray arcs.
    const { solid } = this.visibleStorms();
    let i = 0;
    for (const storm of solid) {
      const front = this.frontOf(storm, this.state.step);
      const at = front === null ? null : positionAt(storm, front);
      if (at === null || i >= NEAR_MAX) continue;
      near[i++]?.set(at.lon, at.lat);
    }
    count.value = i;
  }

  /**
   * Draw one frame. With `burning` false the heat passes are skipped: the still that
   * reduced motion and the fallback image show.
   */
  render(now: number, dt: number, burning: boolean): void {
    // Wrapped so the shaders' hash stays precise however long the page is open.
    const time = (now / 1000) % 97;
    const surface = this.surfaceUniforms;
    const uStep = surface["uStep"];
    const uAtlas = surface["uAtlas"];
    if (uStep) uStep.value = Math.min(this.state.step, this.lastStep);
    if (uAtlas) uAtlas.value = this.atlases[this.state.run];
    this.placeNear();
    const dim = this.compositeUniforms["uDim"];
    if (dim) {
      const toward = this.state.focus === null ? 0 : 1;
      const current = dim.value as number;
      dim.value = burning ? current + (toward - current) * Math.min(1, dt * 12) : toward;
    }

    const r = this.renderer;
    r.setRenderTarget(this.reliefTarget);
    r.render(this.reliefScene, this.camera);
    this.buildMarks(burning, now);
    r.setRenderTarget(this.marksTarget);
    r.render(this.marksScene, this.camera);

    const [heatA, heatB] = this.heatTargets;
    if (burning) {
      const source = this.heatIndex === 0 ? heatA : heatB;
      const sink = this.heatIndex === 0 ? heatB : heatA;
      const heat = this.heatUniforms;
      if (heat["uPrevious"]) heat["uPrevious"].value = source.texture;
      if (heat["uTime"]) heat["uTime"].value = time;
      // Frame-rate independent cooling, quick: the fire's shape comes from its stamps, and
      // the heat field only smears and lifts it for a tenth of a second.
      if (heat["uDecay"]) heat["uDecay"].value = Math.exp(-Math.min(dt, 0.1) * 7);
      const stampTime = this.stampUniforms["uTime"];
      if (stampTime) stampTime.value = time;
      r.setRenderTarget(sink);
      r.render(this.heatScene, this.flat);
      this.buildStamps(now, true);
      r.autoClear = false;
      r.render(this.stampScene, this.camera);
      r.autoClear = true;
      this.heatIndex = 1 - this.heatIndex;
    } else {
      // A still frame carries the same ribbons, stamped once without the heat's drift.
      this.clearHeat();
      this.buildStamps(now, false);
      r.setRenderTarget(this.heatIndex === 0 ? heatA : heatB);
      r.autoClear = false;
      r.render(this.stampScene, this.camera);
      r.autoClear = true;
    }
    const composite = this.compositeUniforms;
    const [heatNow] = this.heatIndex === 0 ? [heatA, heatB] : [heatB, heatA];
    if (composite["uHeat"]) composite["uHeat"].value = heatNow.texture;
    if (composite["uTime"]) composite["uTime"].value = time;
    const [outA, outB] = this.outTargets;
    const [outNow, outBefore] = this.outIndex === 0 ? [outA, outB] : [outB, outA];
    r.setRenderTarget(outNow);
    r.render(this.compositeScene, this.flat);

    const blit = this.blitUniforms;
    const progress = this.dissolveStart < 0 ? 1 : Math.min((now - this.dissolveStart) / 1100, 1);
    if (progress >= 1) this.dissolveStart = -1;
    if (blit["uNow"]) blit["uNow"].value = outNow.texture;
    if (blit["uBefore"]) blit["uBefore"].value = outBefore.texture;
    if (blit["uDissolve"]) blit["uDissolve"].value = burning ? progress : 1;
    r.setRenderTarget(null);
    r.render(this.blitScene, this.flat);
  }

  dispose(): void {
    for (const t of [
      this.reliefTarget,
      this.marksTarget,
      ...this.heatTargets,
      ...this.outTargets,
    ]) {
      t.dispose();
    }
    for (const texture of this.textures) texture.dispose();
    for (const scene of [
      this.reliefScene,
      this.marksScene,
      this.stampScene,
      this.heatScene,
      this.compositeScene,
      this.blitScene,
    ]) {
      scene.traverse((object) => {
        if (object instanceof Mesh || object instanceof Points) {
          (object.geometry as BufferGeometry).dispose();
          (object.material as ShaderMaterial).dispose();
        }
      });
    }
    this.renderer.dispose();
  }
}
