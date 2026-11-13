import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { expect, test } from "@playwright/test";

const FIXTURE = new URL("../../public/experiments/demo-fixture/", import.meta.url);
const OFFSETS_KM = [-300, -150, 0, 150, 300, 450, 600] as const;
const MEMBERS = 8;
const STEPS = 5;
const LANDFALL_STEP = 4;

interface Stop {
  readonly id: string;
  readonly direction: "east" | "west" | "none";
  readonly distance_km: number;
  readonly tracks: string;
}

const STOPS: readonly Stop[] = OFFSETS_KM.map((offset) => {
  const direction = offset === 0 ? "none" : offset < 0 ? "west" : "east";
  const id = `${direction}_${String(Math.abs(offset))}`;
  return { id, direction, distance_km: Math.abs(offset), tracks: `stops/${id}/tracks.json` };
});

function tracksFor(offset: number): readonly object[] {
  const runs = ["baseline", "fork", "control"] as const;
  return runs.flatMap((run) =>
    Array.from({ length: MEMBERS * STEPS }, (_, index) => {
      const member = Math.floor(index / STEPS);
      const step = index % STEPS;
      const moved = run === "fork" ? offset / 100 : 0;
      return { run, member, step, lat: 28 + step + member * 0.2 + moved, lon: -75 + step };
    }),
  );
}

test.use({ viewport: { width: 900, height: 900 } });

test("plots one column of eight member points for each published stop", async ({ page }) => {
  const manifest: unknown = {
    ...(JSON.parse(readFileSync(new URL("manifest.json", FIXTURE), "utf8")) as object),
    id: "sweep-fixture",
    title: "Sweep fixture: synthetic tracks",
    kind: "sweep",
    sweep: { name: "displacement", baseline: "none_0", landfall_step: LANDFALL_STEP, stops: STOPS },
  };
  await page.route("**/experiments/sweep-fixture/**", async (route) => {
    const path = new URL(route.request().url()).pathname.split("/experiments/sweep-fixture/")[1];
    const index = STOPS.findIndex((candidate) => candidate.tracks === path);
    if (path === "manifest.json") {
      await route.fulfill({ json: manifest });
    } else if (path === "tracks.json" || index >= 0) {
      await route.fulfill({ json: tracksFor(OFFSETS_KM[Math.max(index, 0)] ?? 0) });
    } else if (path !== undefined && existsSync(new URL(path, FIXTURE))) {
      await route.fulfill({ path: fileURLToPath(new URL(path, FIXTURE)) });
    } else {
      await route.fulfill({ status: 404 });
    }
  });

  await page.goto("/#/e/sweep-fixture");
  const columns = page.locator("#sweep-chart g[data-stop]");
  await expect(columns).toHaveCount(STOPS.length);
  for (const stop of STOPS) {
    await expect(page.locator(`#sweep-chart g[data-stop="${stop.id}"] circle`)).toHaveCount(
      MEMBERS,
    );
  }
  for (const label of await page.locator("#sweep-chart text").all()) {
    await expect(label).toHaveCSS("font-size", "10px");
    await expect(label).toHaveCSS("fill", "rgb(236, 230, 214)");
  }
  await page.locator("#sheet-toggle").click();
  await page.locator("#displacement").fill("4");
  await expect(page).toHaveURL(/stop=east_300/);
  await page.reload();
  await page.locator("#sheet-toggle").click();
  await expect(page.locator("#displacement")).toHaveValue("4");
  await expect(page.locator("#sweep-chart-readout")).toContainText(
    `across ${String(MEMBERS)} members`,
  );
});
