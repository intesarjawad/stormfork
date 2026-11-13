import { readFileSync } from "node:fs";
import { expect, test, type Page } from "@playwright/test";

declare global {
  interface Window {
    dialRegistrations?: { id: string; type: string; signal: AbortSignal | undefined }[];
  }
}

test.use({ viewport: { width: 390, height: 844 }, reducedMotion: "reduce" });

async function landingFixture(page: Page): Promise<void> {
  await page.route("**/experiments/sandy_east_300km/**", async (route) => {
    const name = new URL(route.request().url()).pathname.split("sandy_east_300km/")[1];
    if (name === "divergence.json") {
      await route.fulfill({ json: { steps: [] } });
    } else if (name === "reproductions.json") {
      await route.fulfill({ json: [] });
    } else {
      await route.fulfill({ path: `public/experiments/demo-fixture/${name ?? ""}` });
    }
  });
}

test("visible atlas playback advances, pauses and stops on route exit", async ({ page }) => {
  await page.clock.install();
  await page.goto("/#/e/demo-fixture");
  await expect(page.locator("#globe-panel canvas")).toBeVisible();
  await page.locator("#sheet-toggle").click();
  const play = page.locator("#controls button[aria-pressed]");
  await play.click();
  await expect(play).toHaveAttribute("aria-pressed", "true");
  await page.clock.runFor(1000);
  await expect(page.locator("#frame")).not.toHaveValue("0");
  await play.click();
  await expect(play).toHaveAttribute("aria-pressed", "false");
  const frame = await page.locator("#frame").inputValue();
  await page.clock.runFor(1000);
  await expect(page.locator("#frame")).toHaveValue(frame);
  await play.click();
  await page.locator('[data-nav="how"]').click();
  await expect(play).toHaveAttribute("aria-pressed", "false");
  expect(await page.evaluate(() => window.stormforkGlobe)).toBeUndefined();
});

test("unknown score towns cannot expand the page", async ({ page }) => {
  await page.route("**/scores/scores.json", (route) =>
    route.fulfill({ path: "tests/fixtures/scores.json" }),
  );
  for (const town of ["x".repeat(100), "ZZZZ"]) {
    await page.goto(`/#/scores/${town}`);
    await expect(page.locator("#scoreboard-status")).toContainText("Unknown town");
    await expect(page.locator("#scoreboard-table tbody tr")).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      390,
    );
  }
});

test("scores identify their metric and restrict intervals to WN2 columns", async ({ page }) => {
  await page.route("**/scores/scores.json", (route) =>
    route.fulfill({ path: "tests/fixtures/scores.json" }),
  );
  await page.goto("/#/scores/KMIA");
  await expect(page.locator("#scoreboard-caption")).toContainText("Mean absolute error (MAE)");
  expect(
    await page
      .locator("#scoreboard-caption")
      .evaluate((node) => node.getBoundingClientRect().right),
  ).toBeLessThanOrEqual(390);
  await expect(page.locator("#scoreboard-variable option").first()).toHaveText("2 m temperature");
  const cells = page.locator("#scoreboard-table tbody td");
  await expect(cells.nth(0)).not.toContainText("WN2 − IFS");
  await expect(cells.nth(1)).toContainText("WN2 − IFS interval");
  await expect(cells.nth(2)).toContainText("WN2 − IFS interval");
});

test("geopotential label matches the field's published units", async ({ page }) => {
  const manifest = JSON.parse(
    readFileSync("public/experiments/demo-fixture/manifest.json", "utf8"),
  ) as { fields: unknown[] };
  manifest.fields.push({
    name: "z500",
    files: ["mslp"],
    range: [48000, 59000],
    encoding: "linear",
    resolution_deg: 1,
  });
  await page.route("**/demo-fixture/manifest.json", (route) => route.fulfill({ json: manifest }));
  await page.goto("/#/e/demo-fixture?layer=z500");
  await expect(page.locator("#field-legend")).toContainText("500 hPa geopotential");
  await expect(page.locator("#field-legend")).toContainText("48000 · 59000 m²/s²");
  await expect(page.locator("#tracks-section")).toContainText("fork members are dashed");
  await expect(page.locator("#tracks-section")).toContainText("fork means carry diamond marks");
});

test("returning home disposes every previous dial and run listener", async ({ page }) => {
  await landingFixture(page);
  await page.goto("/#/");
  await expect(page.locator('#hero[data-state="ready"][data-relief="ready"]')).toBeVisible();
  await expect(page.locator("#dial")).toHaveAttribute("aria-valuenow", /\d+/);
  const active = await page.evaluate(() => {
    const registrations: { id: string; type: string; signal: AbortSignal | undefined }[] = [];
    // Binding would attach listeners to the prototype instead of each monitored element.
    // eslint-disable-next-line @typescript-eslint/unbound-method
    const original = EventTarget.prototype.addEventListener;
    EventTarget.prototype.addEventListener = function (type, callback, options) {
      if (
        this instanceof Element &&
        ["dial", "run-baseline", "run-fork", "landing-play"].includes(this.id)
      ) {
        registrations.push({
          id: this.id,
          type,
          signal: typeof options === "object" ? options.signal : undefined,
        });
      }
      original.call(this, type, callback, options);
    };
    Object.assign(window, { dialRegistrations: registrations });
    return registrations.length;
  });
  expect(active).toBe(0);
  for (let visit = 0; visit < 2; visit += 1) {
    await page.locator('[data-nav="how"]').click();
    await page.locator('[data-nav="home"]').click();
    await expect
      .poll(() => page.evaluate(() => (window.dialRegistrations ?? []).length))
      .toBe((visit + 1) * 7);
    await expect(page.locator('#hero[data-state="ready"][data-relief="ready"]')).toBeVisible();
  }
  const listeners = await page.evaluate(() => {
    const entries = window.dialRegistrations ?? [];
    return entries.map(({ id, type, signal }) => ({ id, type, aborted: signal?.aborted }));
  });
  expect(listeners.length).toBeGreaterThanOrEqual(14);
  expect(listeners.every((listener) => listener.aborted !== undefined)).toBe(true);
  for (const listener of listeners.slice(0, 7)) expect(listener.aborted).toBe(true);
  await page.locator("#dial").focus();
  await page.keyboard.press("Home");
  await expect(page.locator("#dial")).toHaveAttribute("aria-valuenow", "0");
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("#dial")).toHaveAttribute("aria-valuenow", "1");
});

test("main reserves the viewport while route code loads", async ({ page }) => {
  let release: () => void = () => undefined;
  const ready = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route("**/src/main.ts", async (route) => {
    await ready;
    await route.continue();
  });
  try {
    await page.goto("/#/how", { waitUntil: "commit" });
    await expect
      .poll(async () => page.locator("footer").evaluate((node) => node.getBoundingClientRect().top))
      .toBeGreaterThanOrEqual(844);
  } finally {
    release();
  }
  await expect(page.locator("#how")).toBeVisible();
});

test("ledger navigation retains its type and rules at both viewport sizes", async ({ page }) => {
  await page.goto("/#/how");
  await page.evaluate(() => document.fonts.ready);
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: width === 1440 ? 900 : 844 });
    await expect(page.locator(".bar")).toHaveCSS("border-bottom-style", "double");
    await expect(page.locator(".bar")).toHaveCSS("padding-left", width === 1440 ? "32px" : "16px");
    await expect(page.locator(".mark")).toHaveCSS("font-size", width === 1440 ? "21px" : "18px");
    await expect(page.locator("#how h2")).toHaveCSS("font-stretch", "100%");
    await expect(page.locator('[data-nav="how"]')).toHaveCSS("font-style", "normal");
    await expect(page.locator('[data-nav="scores"]')).toHaveCSS("font-style", "italic");
    expect(
      await page
        .locator('[data-nav="scores"]')
        .evaluate((node) => getComputedStyle(node, "::before").content),
    ).toBe('"·"');
  }
});

test("recording overlay retains its layout and is removed on route exit", async ({ page }) => {
  await page.clock.install();
  await page.goto("/#/e/demo-fixture?record=1");
  const overlay = page.locator("#record-still");
  await expect(overlay).toBeVisible();
  await expect(overlay).toHaveCSS("position", "absolute");
  await expect(overlay).toHaveCSS("display", "flex");
  await expect(overlay.locator("p")).toHaveCSS("font-size", "20px");
  await expect(page.locator("header")).toBeHidden();
  await page.goto("/#/how");
  await expect(overlay).toHaveCount(0);
  await expect(page.locator("html")).not.toHaveAttribute("data-record");
  await expect(page.locator("header")).toBeVisible();
});

test("navigation fits while bundled fonts are unavailable", async ({ page }) => {
  await page.route("**/*.woff2", (route) => route.abort());
  await page.goto("/#/how");
  await expect(page.locator("#how")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  for (const link of await page.locator(".bar nav a").all()) {
    const right = await link.evaluate((node) => node.getBoundingClientRect().right);
    expect(right).toBeLessThanOrEqual(390);
  }
});

for (const face of ["normal", "italic"]) {
  test(`partial ${face} font loading keeps error routes within the phone viewport`, async ({
    page,
  }) => {
    await page.route(`**/im-fell-english-latin-${face}*.woff2`, (route) => route.abort());
    await page.goto(`/#/e/${"x".repeat(100)}`);
    await expect(page.locator("#stage-state")).toContainText("not in the build");
    await page.evaluate(() => document.fonts.ready);
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      390,
    );
  });
}

test("each route exposes its own page and disposes the atlas", async ({ page }) => {
  await landingFixture(page);
  await page.route("**/scores/scores.json", (route) =>
    route.fulfill({ path: "tests/fixtures/scores.json" }),
  );
  const routes: readonly (readonly [string, string])[] = [
    ["#/e/demo-fixture", "atlas"],
    ["#/how", "how"],
    ["#/scores", "scoreboard"],
    ["#/card", "landing"],
    ["#/", "landing"],
    ["#/unknown", "missing"],
  ];
  for (const [hash, section] of routes) {
    await page.goto(`/${hash}`);
    await expect(page.locator(`#${section}`)).toBeVisible();
    const visible = await page
      .locator("#main > section:visible")
      .evaluateAll((nodes) => nodes.map((node) => node.id));
    expect(visible).toEqual([section]);
  }
  expect(await page.evaluate(() => window.stormforkGlobe)).toBeUndefined();
});

test("score-to-atlas errors stay within the phone viewport before the next paint", async ({
  page,
}) => {
  await page.route("**/*.woff2", (route) => route.abort());
  await page.route("**/scores/scores.json", (route) =>
    route.fulfill({ path: "tests/fixtures/scores.json" }),
  );
  for (let visit = 0; visit < 20; visit += 1) {
    await page.goto("/#/scores");
    await expect(page.locator("#scoreboard-table tbody tr").first()).toBeVisible();
    await page.goto(`/#/e/${"x".repeat(100)}${String(visit)}`);
    await expect(page.locator("#stage-state")).toContainText("not in the build");
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      390,
    );
  }
});
