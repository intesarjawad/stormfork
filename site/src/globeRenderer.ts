import {
  BufferGeometry,
  Material,
  Object3D,
  MeshBasicMaterial,
  ClampToEdgeWrapping,
  Color,
  ColorManagement,
  DataTexture,
  Float32BufferAttribute,
  LineSegments,
  Mesh,
  NearestFilter,
  LinearSRGBColorSpace,
  NoColorSpace,
  OrthographicCamera,
  PerspectiveCamera,
  RGBAFormat,
  Scene,
  ShaderMaterial,
  SphereGeometry,
  Spherical,
  Texture,
  Vector2,
  Vector3,
  WebGLRenderer,
  WebGLRenderTarget,
  type IUniform,
} from "three";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { ATLAS_COLUMNS, ATLAS_ROWS, FRAME_COUNT } from "./atlas";
import {
  BACKGROUND_COLOUR,
  CONTOUR_COLOUR,
  LUT_SIZE,
  SEA_COLOUR,
  rampLut,
  trackColour,
  type Ramp,
} from "./colours";
import {
  fieldSelection,
  frameSelection,
  SPEED_MAX_MPS,
  textureFiles,
  type FieldSelection,
} from "./fieldLayer";
import { coastlineLayer, markerLayer, outlineLayer, trackLayer } from "./layers";
import type { FieldMetadata } from "./manifest";
import type { TrackPoint } from "./tracks";
import { SURFACE_RADIUS, lonLatToPosition } from "./tracks";

const GLOBE_SEGMENTS_X = 128;
const GLOBE_SEGMENTS_Y = 64;
const CAMERA_FOV_DEG = 45;
const CAMERA_DISTANCE = 3.2;
const MIN_CAMERA_DISTANCE = 1.2;
const MAX_CAMERA_DISTANCE = 8;
const MAX_PIXEL_RATIO = 2;
const DEFAULT_OPACITY = 1;
const CONTOUR_WIDTH_PX = 1.5;
const CONTOUR_STRENGTH = 0.4;
const FIELD_FILL = 0.55;
const PROBE_SIZE = 1;
const PROBE_UV_EPSILON = 1e-6;

export interface GlobeOptions {
  readonly reducedMotion?: boolean;
}

export interface TrackSeries {
  readonly run: string;
  readonly points: readonly TrackPoint[];
  /** Line weight in pixels; an ensemble mean is heavier than a member. */
  readonly weight?: number;
  readonly opacity?: number;
  /** Diamond marks at these positions, the steps of an ensemble mean. */
  readonly markers?: readonly TrackPoint[];
}

const NIGHT_VERTEX = /* glsl */ `
varying vec3 vNormal;
void main() {
  vNormal = normalize(position);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const NIGHT_FRAGMENT = /* glsl */ `
uniform vec3 uSun;
uniform vec3 uGround;
varying vec3 vNormal;
void main() {
  float day = dot(normalize(vNormal), uSun);
  gl_FragColor = vec4(uGround, 0.62 * (1.0 - smoothstep(-0.1, 0.12, day)));
}
`;

const VERTEX_SHADER = /* glsl */ `
varying vec2 vUv;

void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const FRAGMENT_SHADER = /* glsl */ `
uniform sampler2D uAtlas;
uniform sampler2D uAtlasV;
uniform sampler2D uBase;
uniform sampler2D uBaseV;
uniform sampler2D uLut;
uniform int uKind;
uniform vec2 uRange;
uniform vec2 uTile;
uniform int uDiff;
uniform float uFrameA;
uniform float uFrameB;
uniform float uMix;
uniform float uFrames;
uniform float uOpacity;
uniform float uContour;
uniform vec3 uContourColour;
uniform float uFill;
uniform float uStyled;

varying vec2 vUv;

float decodeValue(float encoded) {
  if (uKind == 1) {
    float root = encoded * sqrt(uRange.y);
    return root * root;
  }
  return uRange.x + encoded * (uRange.y - uRange.x);
}

vec2 atlasPosition(float frame) {
  float x = mod(frame, ${String(ATLAS_COLUMNS)}.0) * uTile.x + 0.5 + vUv.x * (uTile.x - 1.0);
  float y = floor(frame / ${String(ATLAS_COLUMNS)}.0) * uTile.y + 0.5
    + clamp(vUv.y * (uTile.y - 1.0), 0.0, uTile.y - 1.0);
  vec2 size = vec2(${String(ATLAS_COLUMNS)}.0 * uTile.x, ${String(ATLAS_ROWS)}.0 * uTile.y);
  return vec2(x, y) / size;
}

float scalarAt(sampler2D atlas, float frame) {
  return decodeValue(texture2D(atlas, atlasPosition(frame)).r);
}

vec2 windAt(sampler2D atlas, sampler2D atlasV, float frame) {
  vec2 position = atlasPosition(frame);
  return vec2(
    decodeValue(texture2D(atlas, position).r),
    decodeValue(texture2D(atlasV, position).r)
  );
}

vec3 lutColour(float rampPosition) {
  float index = floor(clamp(rampPosition, 0.0, 1.0) * ${String(LUT_SIZE - 1)}.0 + 0.5);
  return texture2D(uLut, vec2((index + 0.5) / ${String(LUT_SIZE)}.0, 0.5)).rgb;
}

float contourMask(float value) {
  if (uContour <= 0.0) {
    return 0.0;
  }
  float scaled = value / uContour;
  float lineDistance = abs(fract(scaled + 0.5) - 0.5);
  float width = max(fwidth(scaled), 1.0e-6);
  return 1.0 - smoothstep(0.0, width * ${String(CONTOUR_WIDTH_PX)}, lineDistance);
}

void main() {
  float frameA = clamp(uFrameA, 0.0, uFrames - 1.0);
  float frameB = clamp(uFrameB, 0.0, uFrames - 1.0);
  float span = uRange.y - uRange.x;
  vec3 colour;
  float rampPosition;
  float tint = 0.0;
  float lines = 0.0;

  if (uKind == 2) {
    vec2 wind = mix(
      windAt(uAtlas, uAtlasV, frameA),
      windAt(uAtlas, uAtlasV, frameB),
      uMix
    );
    float speed = length(wind);
    if (uDiff == 1) {
      vec2 base = mix(
        windAt(uBase, uBaseV, frameA),
        windAt(uBase, uBaseV, frameB),
        uMix
      );
      speed -= length(base);
      rampPosition = 0.5 + speed / span;
    } else {
      rampPosition = speed / ${String(SPEED_MAX_MPS)}.0;
    }
    colour = lutColour(rampPosition);
    tint = uDiff == 1 ? clamp(abs(rampPosition - 0.5) * 2.4, 0.0, 1.0) : smoothstep(0.08, 0.9, rampPosition);
  } else {
    float value = mix(scalarAt(uAtlas, frameA), scalarAt(uAtlas, frameB), uMix);
    float contourValue = value;
    if (uDiff == 1) {
      value -= mix(scalarAt(uBase, frameA), scalarAt(uBase, frameB), uMix);
      rampPosition = 0.5 + value / span;
    } else {
      rampPosition = (value - uRange.x) / span;
    }
    colour = lutColour(rampPosition);
    lines = contourMask(contourValue);
    if (uDiff == 1) {
      tint = clamp(abs(rampPosition - 0.5) * 2.4, 0.0, 1.0);
    } else if (uContour <= 0.0) {
      tint = smoothstep(0.08, 0.9, rampPosition);
    }
    colour = mix(colour, uContourColour, lines);
  }

  if (uStyled < 0.5) {
    gl_FragColor = vec4(colour, uOpacity);
    return;
  }
  // The field never covers the ground: isobars in ice at 40 percent, any tint only where it is strong.
  gl_FragColor = vec4(colour, clamp(tint * uFill + lines * ${String(CONTOUR_STRENGTH)}, 0.0, 1.0) * uOpacity);
}
`;

function createLut(ramp: Ramp): DataTexture {
  const texture = new DataTexture(rampLut(ramp), LUT_SIZE, 1, RGBAFormat);
  texture.colorSpace = NoColorSpace;
  texture.minFilter = NearestFilter;
  texture.magFilter = NearestFilter;
  texture.wrapS = ClampToEdgeWrapping;
  texture.wrapT = ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}

function createMaterial(luts: Readonly<Record<Ramp, DataTexture>>): ShaderMaterial {
  const uniforms: Record<string, IUniform> = {
    uAtlas: { value: null },
    uAtlasV: { value: null },
    uBase: { value: null },
    uBaseV: { value: null },
    uLut: { value: luts.scalar },
    uKind: { value: 0 },
    uRange: { value: new Vector2(0, 1) },
    uTile: { value: new Vector2(361, 181) },
    uDiff: { value: 0 },
    uFrameA: { value: 0 },
    uFrameB: { value: 0 },
    uMix: { value: 0 },
    uFrames: { value: FRAME_COUNT },
    uOpacity: { value: DEFAULT_OPACITY },
    uContour: { value: 0 },
    uContourColour: { value: new Color(CONTOUR_COLOUR) },
    uFill: { value: FIELD_FILL },
    uStyled: { value: 1 },
  };
  return new ShaderMaterial({
    uniforms,
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    transparent: true,
  });
}

function setUniform(material: ShaderMaterial, name: string, value: unknown): void {
  const uniform = material.uniforms[name];
  if (uniform === undefined) {
    throw new Error(`field material has no ${name} uniform`);
  }
  uniform.value = value;
}

function textureNamed(
  textures: readonly Texture[],
  field: FieldMetadata,
  name: string | null,
): Texture | null {
  if (name === null) {
    return null;
  }
  const index = field.files.indexOf(name);
  return index < 0 ? null : (textures[index] ?? null);
}

function disposeObject(object: Object3D): void {
  const disposable = object as Object3D & {
    geometry: BufferGeometry;
    material: Material | Material[];
  };
  disposable.geometry.dispose();
  for (const material of Array.isArray(disposable.material)
    ? disposable.material
    : [disposable.material]) {
    material.dispose();
  }
}

function withinUnit(value: number): number {
  return Math.min(Math.max(value, PROBE_UV_EPSILON), 1 - PROBE_UV_EPSILON);
}

/**
 * Draw one experiment field on a unit sphere with its tracks and coastlines.
 *
 * Rendering is on demand: controls, resizes and uniform changes request a
 * frame, so an untouched globe schedules no animation callbacks.
 */
export class GlobeRenderer {
  readonly renderer: WebGLRenderer;
  readonly scene: Scene;
  readonly camera: PerspectiveCamera;
  readonly controls: OrbitControls;
  readonly material: ShaderMaterial;

  readonly #mesh: Mesh<SphereGeometry, ShaderMaterial>;
  readonly #luts: Readonly<Record<Ramp, DataTexture>>;
  readonly #staticLines: LineSegments[];
  readonly #trackObjects: Object3D[] = [];
  readonly #night: Mesh<SphereGeometry, ShaderMaterial>;
  readonly #base: Mesh<SphereGeometry, MeshBasicMaterial>;
  readonly #reducedMotion: boolean;

  #frameRequest: number | null = null;
  #field: FieldMetadata | null = null;
  #textures: ReadonlyMap<string, readonly Texture[]> | null = null;
  #run = "baseline";
  #difference = false;
  #frame = 0;
  #opacity = DEFAULT_OPACITY;
  #width = 1;
  #height = 1;

  constructor(canvas: HTMLCanvasElement, options: GlobeOptions = {}) {
    // Ramp and layer colours are authored as sRGB bytes. Disabling colour
    // management keeps a LUT entry's bytes equal to the pixel on screen and to
    // the value a probe reads back, which is what the ramps were chosen for.
    ColorManagement.enabled = false;
    this.#reducedMotion = options.reducedMotion ?? false;

    this.renderer = new WebGLRenderer({ canvas, antialias: true });
    // The shaders write display colours themselves, so the renderer must not convert them again.
    this.renderer.outputColorSpace = LinearSRGBColorSpace;
    this.renderer.setClearColor(BACKGROUND_COLOUR, 1);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, MAX_PIXEL_RATIO));

    this.scene = new Scene();
    this.camera = new PerspectiveCamera(CAMERA_FOV_DEG, 1, 0.1, 20);
    this.camera.position.set(0, 0, CAMERA_DISTANCE);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = !this.#reducedMotion;
    this.controls.autoRotate = false;
    this.controls.enablePan = false;
    this.controls.minDistance = MIN_CAMERA_DISTANCE;
    this.controls.maxDistance = MAX_CAMERA_DISTANCE;
    this.controls.target.set(0, 0, 0);
    canvas.addEventListener("keydown", (event) => {
      const position = new Spherical().setFromVector3(
        this.camera.position.clone().sub(this.controls.target),
      );
      const angle = Math.PI / 18;
      switch (event.key) {
        case "ArrowLeft":
          position.theta -= angle;
          break;
        case "ArrowRight":
          position.theta += angle;
          break;
        case "ArrowUp":
          position.phi -= angle;
          break;
        case "ArrowDown":
          position.phi += angle;
          break;
        case "+":
        case "=":
          position.radius /= 1.2;
          break;
        case "-":
        case "_":
          position.radius *= 1.2;
          break;
        default:
          return;
      }
      event.preventDefault();
      position.makeSafe();
      position.radius = Math.max(
        MIN_CAMERA_DISTANCE,
        Math.min(MAX_CAMERA_DISTANCE, position.radius),
      );
      this.camera.position.setFromSpherical(position).add(this.controls.target);
      this.controls.update();
      this.requestRender();
    });
    this.controls.addEventListener("change", () => {
      this.requestRender();
    });

    this.#luts = {
      scalar: createLut("scalar"),
      speed: createLut("speed"),
      diverging: createLut("diverging"),
    };
    this.material = createMaterial(this.#luts);
    this.#mesh = new Mesh(
      new SphereGeometry(SURFACE_RADIUS, GLOBE_SEGMENTS_X, GLOBE_SEGMENTS_Y),
      this.material,
    );
    this.#mesh.renderOrder = 0;
    this.#base = new Mesh(
      new SphereGeometry(SURFACE_RADIUS * 0.998, GLOBE_SEGMENTS_X, GLOBE_SEGMENTS_Y),
      new MeshBasicMaterial({ color: SEA_COLOUR }),
    );
    this.#night = new Mesh(
      new SphereGeometry(SURFACE_RADIUS * 1.002, GLOBE_SEGMENTS_X, GLOBE_SEGMENTS_Y),
      new ShaderMaterial({
        uniforms: {
          uSun: { value: new Vector3(1, 0, 0) },
          uGround: { value: new Color(BACKGROUND_COLOUR) },
        },
        vertexShader: NIGHT_VERTEX,
        fragmentShader: NIGHT_FRAGMENT,
        transparent: true,
        depthWrite: false,
      }),
    );
    this.#night.renderOrder = 0.5;
    this.#staticLines = [outlineLayer(), coastlineLayer()];
    this.scene.add(this.#base, this.#mesh, this.#night);
    this.scene.add(...this.#staticLines);

    this.resize(canvas.clientWidth, canvas.clientHeight);
  }

  get field(): FieldMetadata | null {
    return this.#field;
  }

  get frame(): number {
    return this.#frame;
  }

  /** Number of rendered frames; flat while the globe is untouched. */
  get renderedFrames(): number {
    return this.renderer.info.render.frame;
  }

  setTracks(series: readonly TrackSeries[]): void {
    for (const object of this.#trackObjects) {
      this.scene.remove(object);
      disposeObject(object);
    }
    this.#trackObjects.length = 0;
    for (const track of series) {
      const layer = trackLayer(track.points, track.run, track.weight ?? 1, track.opacity ?? 1);
      layer.material.resolution.set(this.#width, this.#height);
      this.#trackObjects.push(layer);
      this.scene.add(layer);
      if (track.markers !== undefined) {
        const marks = markerLayer(track.markers, trackColour(track.run), BACKGROUND_COLOUR);
        this.#trackObjects.push(marks);
        this.scene.add(marks);
      }
    }
    this.requestRender();
  }

  /** Light the globe as the sun lights it at `iso`, UTC: the night side is dimmed. */
  setSun(iso: string): void {
    const at = new Date(iso.endsWith("Z") ? iso : `${iso}Z`);
    if (Number.isNaN(at.getTime())) {
      return;
    }
    const dayOfYear = (at.getTime() - Date.UTC(at.getUTCFullYear(), 0, 0)) / 86_400_000;
    const declination = -23.44 * Math.cos(((dayOfYear + 10) / 365) * 2 * Math.PI);
    const hours = at.getUTCHours() + at.getUTCMinutes() / 60;
    const [x, y, z] = lonLatToPosition(declination, (12 - hours) * 15);
    setUniform(this.#night.material, "uSun", new Vector3(x, y, z).normalize());
    this.requestRender();
  }

  setField(
    field: FieldMetadata,
    textures: ReadonlyMap<string, readonly Texture[]>,
    run: string,
    difference: boolean,
  ): void {
    this.#field = field;
    this.#textures = textures;
    this.#run = run;
    this.#difference = difference;
    this.#bind();
    this.setFrame(this.#frame);
    this.requestRender();
  }

  setRun(run: string): void {
    this.#run = run;
    this.#bind();
    this.requestRender();
  }

  setDifference(difference: boolean): void {
    this.#difference = difference;
    this.#bind();
    this.requestRender();
  }

  setFrame(frame: number): void {
    this.#frame = frame;
    const { a, b, mix } = frameSelection(frame);
    setUniform(this.material, "uFrameA", a);
    setUniform(this.material, "uFrameB", b);
    setUniform(this.material, "uMix", mix);
    this.requestRender();
  }

  setOpacity(opacity: number): void {
    this.#opacity = Math.min(Math.max(opacity, 0), 1);
    setUniform(this.material, "uOpacity", this.#opacity);
    this.requestRender();
  }

  resize(width: number, height: number): void {
    if (width <= 0 || height <= 0) {
      return;
    }
    this.#width = width;
    this.#height = height;
    for (const object of this.#trackObjects) {
      if (object instanceof LineSegments2) {
        object.material.resolution.set(width, height);
      }
    }
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }

  render(): void {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  requestRender(): void {
    if (this.#frameRequest !== null) {
      return;
    }
    this.#frameRequest = requestAnimationFrame(() => {
      this.#frameRequest = null;
      this.render();
    });
  }

  /**
   * Read the field material at one geographic point into a single pixel.
   *
   * The frame and opacity are restored afterwards, so a probe does not change
   * what the reader sees.
   */
  probeField(frame: number, lat: number, lon: number): readonly [number, number, number, number] {
    const previousFrame = this.#frame;
    const previousOpacity = this.#opacity;
    this.setFrame(frame);
    this.setOpacity(1);
    setUniform(this.material, "uStyled", 0);

    const uv: readonly [number, number] = [
      withinUnit((((lon % 360) + 360) % 360) / 360),
      withinUnit((lat + 90) / 180),
    ];
    const geometry = new BufferGeometry();
    geometry.setAttribute(
      "position",
      new Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3),
    );
    geometry.setAttribute("uv", new Float32BufferAttribute([...uv, ...uv, ...uv], 2));
    const probeScene = new Scene();
    probeScene.add(new Mesh(geometry, this.material));
    const camera = new OrthographicCamera(-1, 1, 1, -1, 0.1, 2);
    camera.position.z = 1;
    const target = new WebGLRenderTarget(PROBE_SIZE, PROBE_SIZE);
    const pixels = new Uint8Array(4);
    const previous = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(target);
    this.renderer.render(probeScene, camera);
    this.renderer.readRenderTargetPixels(target, 0, 0, PROBE_SIZE, PROBE_SIZE, pixels);
    this.renderer.setRenderTarget(previous);
    geometry.dispose();
    target.dispose();

    setUniform(this.material, "uStyled", 1);
    this.setFrame(previousFrame);
    this.setOpacity(previousOpacity);
    const [red = 0, green = 0, blue = 0, alpha = 0] = pixels;
    return [red, green, blue, alpha];
  }

  dispose(): void {
    if (this.#frameRequest !== null) {
      cancelAnimationFrame(this.#frameRequest);
      this.#frameRequest = null;
    }
    this.controls.dispose();
    for (const line of [...this.#staticLines, ...this.#trackObjects]) {
      disposeObject(line);
    }
    this.#base.geometry.dispose();
    this.#base.material.dispose();
    this.#night.geometry.dispose();
    this.#night.material.dispose();
    this.#mesh.geometry.dispose();
    this.material.dispose();
    for (const lut of Object.values(this.#luts)) {
      lut.dispose();
    }
    this.renderer.dispose();
  }

  #bind(): void {
    const field = this.#field;
    const textures = this.#textures;
    if (field === null || textures === null) {
      return;
    }
    const selection: FieldSelection = fieldSelection(field, this.#difference);
    const files = textureFiles(field, this.#difference);
    const current = textures.get(this.#run) ?? [];
    const baseline = textures.get("baseline") ?? [];

    setUniform(this.material, "uAtlas", textureNamed(current, field, files.atlas));
    setUniform(this.material, "uAtlasV", textureNamed(current, field, files.atlasV));
    setUniform(this.material, "uBase", textureNamed(baseline, field, files.base));
    setUniform(this.material, "uBaseV", textureNamed(baseline, field, files.baseV));
    setUniform(this.material, "uKind", selection.kind);
    setUniform(this.material, "uRange", new Vector2(selection.range[0], selection.range[1]));
    setUniform(this.material, "uTile", new Vector2(selection.tile[0], selection.tile[1]));
    setUniform(this.material, "uDiff", selection.diff);
    setUniform(this.material, "uLut", this.#luts[selection.ramp]);
    setUniform(this.material, "uContour", selection.contourInterval);
    setUniform(this.material, "uFrames", FRAME_COUNT);
  }
}
