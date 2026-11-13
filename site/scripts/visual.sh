#!/bin/sh
# Screenshot goldens are rendered and compared only in the pinned Playwright
# image: the host's macOS node_modules cannot load there, and Metal and
# SwiftShader differ by more than the per-pixel budget in playwright.config.ts.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
image=mcr.microsoft.com/playwright:v1.64.1-noble
pnpm_version=12.16.1

# The image ships Node without corepack, so pnpm comes from npm. node_modules
# and the pnpm store live in container-only volumes to keep the host tree as is.
docker run --rm --ipc=host \
  -v "$root/..:/w" \
  -v stormfork-site-node-modules:/w/site/node_modules \
  -v stormfork-pnpm-store:/pnpm-store \
  -w /w/site \
  "$image" \
  sh -eu -c '
    npm install --global --no-fund --no-audit "pnpm@$1" >/dev/null
    pnpm install --frozen-lockfile --ignore-scripts --store-dir /pnpm-store
    shift
    exec pnpm exec playwright test tests/visual "$@"
  ' sh "$pnpm_version" "$@"
