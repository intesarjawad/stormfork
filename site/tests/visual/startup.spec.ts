import { expect, test } from "@playwright/test";

declare global {
  interface Window {
    routeLayoutShift?: number;
  }
}

for (const viewport of [
  { width: 1440, height: 900 },
  { width: 390, height: 844 },
]) {
  test(`cold Method startup keeps content stable at ${String(viewport.width)}px`, async ({
    page,
  }) => {
    await page.setViewportSize(viewport);
    await page.addInitScript(() => {
      window.routeLayoutShift = 0;
      new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (
            "value" in entry &&
            typeof entry.value === "number" &&
            !("hadRecentInput" in entry && entry.hadRecentInput)
          ) {
            window.routeLayoutShift = (window.routeLayoutShift ?? 0) + entry.value;
          }
        }
      }).observe({ type: "layout-shift", buffered: true });
    });
    await page.goto("/#/how");
    await expect(page.locator("#how")).toBeVisible();
    await page.evaluate(() => document.fonts.ready);
    await page.evaluate(
      () =>
        new Promise<void>((resolve) =>
          requestAnimationFrame(() =>
            requestAnimationFrame(() => {
              resolve();
            }),
          ),
        ),
    );
    const cls = await page.evaluate(() => window.routeLayoutShift);
    expect(cls).toBeDefined();
    expect(cls).toBeLessThanOrEqual(0.1);
    await test.info().attach("startup-measurements.json", {
      body: JSON.stringify({ viewport, cls }),
      contentType: "application/json",
    });
  });
}
