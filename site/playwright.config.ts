import { defineConfig, devices } from "@playwright/test";

/**
 * Screenshot goldens are produced and compared only in the pinned Playwright
 * image, because Metal and SwiftShader renders differ by more than the
 * per-pixel budget below.
 */
export default defineConfig({
  testDir: "tests/visual",
  forbidOnly: true,
  reporter: "list",
  expect: {
    toHaveScreenshot: { maxDiffPixelRatio: 0.002 },
  },
  use: {
    baseURL: "http://127.0.0.1:4173",
  },
  webServer: {
    command: "pnpm exec vite --host 127.0.0.1 --port 4173 --strictPort",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: !process.env["CI"],
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
