import { defineConfig } from "vitest/config";

export default defineConfig({
  // Relative output keeps the built site usable from any mount path.
  base: "./",
  build: {
    target: "es2022",
    rolldownOptions: { experimental: { chunkModulesOrder: "module-id" } },
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
});
