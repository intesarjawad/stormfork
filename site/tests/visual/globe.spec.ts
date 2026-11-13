import { expect, test, type Page } from "@playwright/test";

test.use({ viewport: { width: 900, height: 1200 } });

const SPEED_ENTRY_213: readonly [number, number, number] = [131, 199, 195];
// The symmetric wind range puts exactly zero halfway between bytes 127 and
// 128, so calm decodes to about 0.33 m/s, one entry above zero.
const SPEED_OUTSIDE: readonly [number, number, number] = [11, 44, 52];
const DIVERGING_CENTRE: readonly [number, number, number] = [23, 44, 48];

async function openDemo(page: Page): Promise<void> {
  // The opening camera move would still be running when a frame is compared or counted.
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/#/e/demo-fixture");
  await expect(page.locator("#globe-panel canvas")).toBeVisible();
  await expect(page.locator("#controls")).toBeVisible();
  await page.waitForFunction(() => window.stormforkGlobe?.field?.name === "mslp");
  await page.waitForFunction(() => (window.stormforkGlobe?.renderedFrames ?? 0) > 0);
}

async function selectLayer(page: Page, name: string): Promise<void> {
  const before = await page.evaluate(() => window.stormforkGlobe?.renderedFrames ?? 0);
  await page.locator("#sheet-toggle").click();
  await page.locator(`#layer + .segments button[data-value="${name}"]`).click();
  await page.waitForFunction(
    (expected: string) => window.stormforkGlobe?.field?.name === expected,
    name,
  );
  await page.waitForFunction(
    (count: number) => (window.stormforkGlobe?.renderedFrames ?? 0) > count,
    before,
  );
  await page.locator("#sheet-toggle").click();
}

async function settle(page: Page): Promise<void> {
  await page.evaluate(
    () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            resolve();
          });
        });
      }),
  );
}

function rgb(pixel: readonly [number, number, number, number] | undefined): readonly number[] {
  if (pixel === undefined) {
    throw new Error("the probe returned nothing");
  }
  return pixel.slice(0, 3);
}

test("frame 0 of the fixture with the pressure layer", async ({ page }) => {
  await openDemo(page);
  await settle(page);
  await expect(page.locator("#globe-panel")).toHaveScreenshot("globe-mslp.png");
});

test("frame 0 of the fixture with the wind layer", async ({ page }) => {
  await openDemo(page);
  await selectLayer(page, "wind10");
  await settle(page);
  await expect(page.locator("#globe-panel")).toHaveScreenshot("globe-wind.png");
});

test("probes the wind speed, the calm region and the difference centre", async ({ page }) => {
  await openDemo(page);
  await selectLayer(page, "wind10");

  const centre = await page.evaluate(() => window.stormforkProbe?.(0, 20, 300));
  expect(rgb(centre)).toEqual(SPEED_ENTRY_213);

  const outside = await page.evaluate(() => window.stormforkProbe?.(0, 0, 0));
  expect(rgb(outside)).toEqual(SPEED_OUTSIDE);

  await page.locator("#sheet-toggle").click();
  await page.locator('#run + .segments button[data-value="baseline"]').click();
  await page.locator("#difference").check();
  const difference = await page.evaluate(() => window.stormforkProbe?.(0, 20, 300));
  expect(rgb(difference)).toEqual(DIVERGING_CENTRE);
});

test("stops drawing without input", async ({ page }) => {
  await openDemo(page);
  await settle(page);
  const before = await page.evaluate(() => window.stormforkGlobe?.renderedFrames ?? 0);
  const callbacks = await page.evaluate(async () => {
    const original = window.requestAnimationFrame.bind(window);
    let count = 0;
    window.requestAnimationFrame = (callback: FrameRequestCallback): number => {
      count += 1;
      return original(callback);
    };
    await new Promise<void>((resolve) => {
      setTimeout(resolve, 1000);
    });
    window.requestAnimationFrame = original;
    return count;
  });
  const after = await page.evaluate(() => window.stormforkGlobe?.renderedFrames ?? 0);
  expect(callbacks).toBe(0);
  expect(after).toBe(before);
});

test("shows the chart link when WebGL is unavailable", async ({ page }) => {
  await page.addInitScript(() => {
    // The shim needs getContext itself, and binding it would change its `this`.
    // eslint-disable-next-line @typescript-eslint/unbound-method
    const original = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (
      this: HTMLCanvasElement,
      contextId: string,
    ): RenderingContext | null {
      if (contextId === "webgl2" || contextId === "webgl" || contextId === "experimental-webgl") {
        return null;
      }
      return original.call(this, contextId);
    } as typeof original;
  });
  await page.route("**/demo-fixture/tracks.json", (route) => route.fulfill({ json: [] }));
  await page.goto("/#/e/demo-fixture");
  const notice = page.locator("#fallback .notice");
  await expect(notice).toBeVisible();
  await expect(page.locator("#stage-state")).toBeHidden();
  await expect(page.locator("#legend")).toBeHidden();
  await expect(page.locator("#summary")).not.toContainText("static frame");
  await expect(page.locator("#summary")).toBeHidden();
  await expect(page.locator("#notice-link")).toBeHidden();
  await expect(page.locator("#tracks-section")).toBeHidden();
  await expect(notice).not.toContainText("track notes");
  await expect(notice.getByRole("link", { name: "method" })).toHaveAttribute("href", "#/how");
  await expect(notice.getByRole("link", { name: "scores" })).toHaveAttribute("href", "#/scores");
  await expect(notice.getByRole("link", { name: "releases page" })).toHaveAttribute(
    "href",
    /releases$/,
  );
});
