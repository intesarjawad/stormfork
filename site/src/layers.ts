import {
  BufferGeometry,
  Color,
  Float32BufferAttribute,
  LineBasicMaterial,
  LineSegments,
  Points,
  ShaderMaterial,
  Vector2,
} from "three";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { COASTLINES } from "./coastlines";
import { COASTLINE_COLOUR, OUTLINE_COLOUR, trackColour } from "./colours";
import { dashSegments, dashStyle } from "./dash";
import { lonLatToPosition, type TrackPoint, type Vec3 } from "./tracks";

export const COASTLINE_RADIUS = 1.004;
export const OUTLINE_RADIUS = 1.001;
export const TRACK_RADIUS = 1.012;

const OUTLINE_STEP_DEG = 5;
const OUTLINE_DASH = 0.02;
const OUTLINE_GAP = 0.035;

function lineGeometry(segments: readonly Vec3[]): BufferGeometry {
  const geometry = new BufferGeometry();
  const positions = segments.flatMap((point) => [...point]);
  geometry.setAttribute("position", new Float32BufferAttribute(positions, 3));
  return geometry;
}

function lineSegments(segments: readonly Vec3[], colour: string, opacity = 1): LineSegments {
  const material = new LineBasicMaterial({
    color: colour,
    // Keeping every overlay in the transparent pass lets the globe write depth
    // first, so only the near half of a track or coastline is visible.
    transparent: true,
    opacity,
    depthWrite: false,
  });
  const layer = new LineSegments(lineGeometry(segments), material);
  layer.renderOrder = 1;
  return layer;
}

function positions(points: readonly TrackPoint[], radius: number): Vec3[] {
  return points.map((point) => lonLatToPosition(point.lat, point.lon, radius));
}

function pairSegments(points: readonly Vec3[]): readonly Vec3[] {
  return dashSegments(points, 0, 0);
}

/**
 * A track as the plate draws it: `width` in pixels, so an ensemble mean reads heavier than a
 * member. The material's resolution follows the canvas through `setResolution`.
 */
export function trackLayer(
  points: readonly TrackPoint[],
  run: string,
  width = 1,
  opacity = 1,
): LineSegments2 {
  const style = dashStyle(run);
  const sphere = positions(points, TRACK_RADIUS);
  const segments =
    style.dash > 0 ? dashSegments(sphere, style.dash, style.gap) : pairSegments(sphere);
  const geometry = new LineSegmentsGeometry();
  geometry.setPositions(segments.flatMap((point) => [...point]));
  const material = new LineMaterial({
    color: trackColour(run),
    linewidth: width,
    transparent: true,
    opacity,
    depthWrite: false,
  });
  material.resolution = new Vector2(1, 1);
  const layer = new LineSegments2(geometry, material);
  layer.renderOrder = 2;
  return layer;
}

const MARKER_VERTEX = /* glsl */ `
uniform float uSize;
void main() {
  gl_PointSize = uSize;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const MARKER_FRAGMENT = /* glsl */ `
uniform vec3 uColour;
uniform vec3 uGround;
void main() {
  vec2 p = abs(gl_PointCoord - 0.5) * 2.0;
  float d = p.x + p.y;
  if (d > 1.0) { discard; }
  gl_FragColor = vec4(d > 0.7 ? uGround : uColour, 1.0);
}
`;

/** Diamond marks for the ensemble mean's steps, ringed in the ground colour. */
export function markerLayer(
  points: readonly TrackPoint[],
  colour: string,
  ground: string,
  size = 11,
): Points {
  const geometry = new BufferGeometry();
  geometry.setAttribute(
    "position",
    new Float32BufferAttribute(
      positions(points, TRACK_RADIUS + 0.002).flatMap((point) => [...point]),
      3,
    ),
  );
  const material = new ShaderMaterial({
    uniforms: {
      uSize: { value: size },
      uColour: { value: new Color(colour) },
      uGround: { value: new Color(ground) },
    },
    vertexShader: MARKER_VERTEX,
    fragmentShader: MARKER_FRAGMENT,
    transparent: true,
    depthWrite: false,
  });
  const layer = new Points(geometry, material);
  layer.renderOrder = 3;
  return layer;
}

export function coastlineLayer(): LineSegments {
  const segments = COASTLINES.flatMap((line) =>
    pairSegments(
      positions(
        line.map(([lon, lat]) => ({ lat, lon })),
        COASTLINE_RADIUS,
      ),
    ),
  );
  return lineSegments(segments, COASTLINE_COLOUR, 0.9);
}

function greatCircle(points: readonly TrackPoint[]): readonly Vec3[] {
  return dashSegments(positions(points, OUTLINE_RADIUS), OUTLINE_DASH, OUTLINE_GAP);
}

/** Principal great circles that keep the sphere's orientation readable. */
export function outlineLayer(): LineSegments {
  const equator = Array.from({ length: 360 / OUTLINE_STEP_DEG + 1 }, (_, index) => ({
    lat: 0,
    lon: index * OUTLINE_STEP_DEG,
  }));
  const meridian = (lon: number): TrackPoint[] =>
    Array.from({ length: 180 / OUTLINE_STEP_DEG + 1 }, (_, index) => ({
      lat: -90 + index * OUTLINE_STEP_DEG,
      lon,
    }));
  const segments = [
    ...greatCircle(equator),
    ...greatCircle(meridian(0)),
    ...greatCircle(meridian(90)),
  ];
  return lineSegments(segments, OUTLINE_COLOUR, 0.8);
}
