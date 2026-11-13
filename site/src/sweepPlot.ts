import { trackColour } from "./colours";
import { svgNode } from "./dom";
import type { SweepChartLayout } from "./sweepChart";

export const CHART_POINT_RADIUS = 3;
export const CHART_SELECTED_RADIUS = 5;

export function drawSweepChart(svg: SVGSVGElement, layout: SweepChartLayout): void {
  const { plot } = layout;
  const ink = "#9bbab6";
  const nodes: SVGElement[] = [];
  if (layout.band.length > 0) {
    nodes.push(
      svgNode("polygon", {
        points: layout.band,
        fill: trackColour("control"),
        "fill-opacity": "0.25",
        stroke: trackColour("control"),
        "stroke-dasharray": "4 3",
      }),
    );
  }
  nodes.push(
    svgNode("line", {
      x1: String(plot.left),
      y1: String(plot.bottom),
      x2: String(plot.right),
      y2: String(plot.bottom),
      stroke: ink,
    }),
    svgNode("line", {
      x1: String(plot.left),
      y1: String(plot.top),
      x2: String(plot.left),
      y2: String(plot.bottom),
      stroke: ink,
    }),
  );
  for (const tick of layout.yTicks) {
    nodes.push(
      svgNode(
        "text",
        {
          x: String(plot.left - 6),
          y: String(tick.position),
          "text-anchor": "end",
          "dominant-baseline": "middle",
        },
        tick.label,
      ),
    );
  }
  for (const tick of layout.xTicks) {
    nodes.push(
      svgNode(
        "text",
        {
          x: String(tick.position),
          y: String(plot.bottom + 12),
          "text-anchor": "middle",
        },
        tick.label,
      ),
    );
  }
  nodes.push(
    svgNode(
      "text",
      {
        x: String((plot.left + plot.right) / 2),
        y: String(layout.height - 4),
        "text-anchor": "middle",
      },
      "Storm displacement, km (east or north positive)",
    ),
    svgNode(
      "text",
      {
        x: "10",
        y: String((plot.top + plot.bottom) / 2),
        "text-anchor": "middle",
        transform: `rotate(-90 10 ${String((plot.top + plot.bottom) / 2)})`,
      },
      "Landfall latitude, °N",
    ),
  );
  for (const tick of layout.xTicks) {
    const column = svgNode("g", { "data-stop": tick.stop });
    for (const point of layout.points.filter((candidate) => candidate.stop === tick.stop)) {
      const circle = svgNode("circle", {
        cx: point.x.toFixed(2),
        cy: point.y.toFixed(2),
        r: String(CHART_POINT_RADIUS),
        fill: trackColour("fork"),
        stroke: "#061719",
        "stroke-width": "1",
        "data-member": String(point.member),
      });
      column.append(circle);
    }
    nodes.push(column);
  }
  svg.replaceChildren(...nodes);
}
