import { defineConfig } from "@playwright/test";
import base from "./playwright.config";

const url = "http://127.0.0.1:4174";

export default defineConfig({
  ...base,
  use: { ...base.use, baseURL: url },
  webServer: {
    command: "pnpm exec vite preview --host 127.0.0.1 --port 4174 --strictPort",
    url,
    reuseExistingServer: false,
  },
});
