import { readFileSync } from "node:fs";

import { expect, test } from "@playwright/test";

test.use({ viewport: { width: 390, height: 844 }, reducedMotion: "reduce" });

test("short runs clamp navigation, preserve the loaded field on failure and share difference", async ({
  page,
}) => {
  await page.route("**/demo-fixture/manifest.json", async (route) => {
    const response = await route.fetch();
    const manifest = (await response.json()) as Record<string, unknown>;
    manifest["steps"] = 1;
    await route.fulfill({ response, json: manifest });
  });
  await page.route("**/demo-fixture/fork/v10.png", (route) => route.abort());
  await page.goto("/#/e/demo-fixture?step=40");
  await expect(page.locator("#frame")).toHaveValue("1");
  await page.locator("#sheet-toggle").click();
  await page.locator('#layer + .segments button[data-value="wind10"]').click();
  await expect(page.locator("#summary")).toContainText("previous field remains visible");
  await page.locator('#run + .segments button[data-value="fork"]').click();
  await expect(page.locator("#summary")).toContainText("Sea-level pressure");
  await page.locator("#difference").check();
  await expect(page.locator("#field-legend")).toContainText("difference from baseline");
  await expect(page.locator("#field-legend")).toContainText("Pa");
  await expect(page).toHaveURL(/difference=1/);
  await page.reload();
  await page.locator("#sheet-toggle").click();
  await expect(page.locator("#difference")).toBeChecked();
});

test("keyboard can skip, open the sheet and control the camera", async ({ page }) => {
  await page.goto("/#/e/demo-fixture");
  const canvas = page.locator("#globe-panel canvas");
  await expect(canvas).toBeVisible();
  await page.locator(".skip").focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("#main")).toBeFocused();
  await expect(page).toHaveURL(/#\/e\/demo-fixture/);
  await expect(page.locator("#control-fields")).toHaveAttribute("inert", "");
  await page.locator("#sheet-toggle").click();
  await expect(page.locator("#control-fields")).not.toHaveAttribute("inert", "");
  const before = await page.evaluate(() => window.stormforkGlobe?.camera.position.toArray());
  await canvas.focus();
  await page.keyboard.press("ArrowLeft");
  await page.keyboard.press("+");
  const after = await page.evaluate(() => window.stormforkGlobe?.camera.position.toArray());
  expect(after).not.toEqual(before);
});

test("phone scores retain headers, sorting and the requested town", async ({ page }) => {
  await page.route("**/scores/scores.json", (route) =>
    route.fulfill({ path: "tests/fixtures/scores.json", contentType: "application/json" }),
  );
  await page.goto("/#/scores/KMIA");
  await expect(page.locator("#scores-retry")).toBeHidden();
  await expect(page.locator("#scoreboard-table tbody tr")).toHaveCount(1);
  await expect(page.locator("#scoreboard-table tbody tr")).toContainText("KMIA");
  const cells = page.locator("#scoreboard-table tbody td");
  await expect(cells.nth(0)).toContainText("1.00 °C · n=100");
  await expect(cells.nth(0)).not.toContainText("WN2 − IFS interval");
  await expect(cells.nth(2)).toContainText("interval: not estimated");
  await expect(cells.nth(1)).toContainText("-0.4 to -0.1 °C");
  await expect(cells.nth(1)).toContainText("beats IFS");
  await expect(page.locator("#scoreboard-table thead button").first()).toBeVisible();
  await expect(page.locator("#scoreboard-lead option").first()).toContainText("days");
  await page.locator("#scoreboard-table thead button").first().click();
});

test("matched comparisons show each model's published sample size", async ({ page }) => {
  const data = JSON.parse(readFileSync("tests/fixtures/scores.json", "utf8")) as {
    cells: {
      town: string;
      model: string;
      mae: number;
      sample_size: number;
      interval: number[] | null;
      badge: string | null;
    }[];
  };
  for (const cell of data.cells) {
    if (cell.town === "KMIA" && (cell.model === "IFS" || cell.model === "WN2 mean")) {
      cell.mae = cell.model === "IFS" ? 0 : 1;
      cell.sample_size = 1;
      cell.interval = null;
      cell.badge = null;
    }
  }
  await page.route("**/scores/scores.json", (route) => route.fulfill({ json: data }));
  await page.goto("/#/scores/KMIA");
  const cells = page.locator("#scoreboard-table tbody td");
  await expect(cells.nth(0)).toContainText("0.00 °C · n=1");
  await expect(cells.nth(1)).toContainText("1.00 °C · n=1");
  await expect(cells.nth(1)).toContainText("not estimated");
});

test("score errors can be retried and long identifiers wrap", async ({ page }) => {
  let requests = 0;
  await page.route("**/scores/scores.json", async (route) => {
    requests += 1;
    if (requests === 1) await route.fulfill({ status: 500, body: "failed" });
    else
      await route.fulfill({ path: "tests/fixtures/scores.json", contentType: "application/json" });
  });
  await page.goto("/#/scores");
  await expect(page.locator("#scores-state")).toContainText("Scores could not be loaded");
  await page.locator("#scores-retry").click();
  await expect(page.locator("#scoreboard-table tbody tr").first()).toBeVisible();
  await page.goto(`/#/e/${"x".repeat(100)}`);
  await expect(page.locator("#stage-state")).toContainText("not in the build");
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});

test("method diagram remains readable and scrollable on a phone", async ({ page }) => {
  await page.goto("/#/how");
  const diagram = page.locator(".diagram-wrap");
  expect(await diagram.evaluate((node) => node.scrollWidth > node.clientWidth)).toBe(true);
  expect(
    await page
      .locator(".hd-note")
      .first()
      .evaluate((node) => node.getBoundingClientRect().height),
  ).toBeGreaterThanOrEqual(14);
});
