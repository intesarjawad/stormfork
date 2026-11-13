// Run after `pnpm --dir site build`; the ignored Sandy atlas must be in site/public/experiments.
import { chromium } from "../site/node_modules/@playwright/test/index.mjs";
import { execFileSync } from "node:child_process";
import { mkdtemp, readFile, rename, rm, stat } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = path.join(root, "docs/demo.gif");
const temporary = await mkdtemp(path.join(os.tmpdir(), "stormfork-demo-"));
const origin = "https://intesarjawad.github.io/stormfork/";
const types: Record<string, string> = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".png": "image/png",
  ".woff2": "font/woff2",
};
const ffmpeg = (...args: string[]) =>
  execFileSync(
    "ffmpeg",
    ["-hide_banner", "-loglevel", "error", "-y", ...args],
    { stdio: "inherit" },
  );
let browser;
try {
  browser = await chromium.launch({
    headless: true,
    executablePath:
      process.env.BRAVE_PATH ??
      "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    args: ["--enable-unsafe-swiftshader"],
  });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 960 },
    recordVideo: { dir: temporary, size: { width: 1280, height: 960 } },
  });
  await context.route(`${origin}**`, async (route) => {
    const relative =
      decodeURIComponent(
        new URL(route.request().url()).pathname.slice("/stormfork/".length),
      ) || "index.html";
    const filename = path.resolve(root, "site/dist", relative);
    if (!filename.startsWith(path.join(root, "site/dist") + path.sep))
      return route.abort();
    try {
      await route.fulfill({
        body: await readFile(filename),
        contentType:
          types[path.extname(filename)] ?? "application/octet-stream",
      });
    } catch {
      await route.abort();
    }
  });
  const page = await context.newPage();
  await page.goto(origin);
  await page.locator('#hero[data-relief="ready"]').waitFor({ timeout: 90_000 });
  const dial = page.locator("#dial");
  const stamp = page.locator("#stamp");
  await dial.focus();
  await page.keyboard.press("Home");
  for (let step = 0; step < 4; step++) await page.keyboard.press("PageUp");
  if (!(await stamp.innerText()).includes("CONFIRMED"))
    throw new Error("The split stamp is not confirmed");
  const box = await stamp.boundingBox();
  if (!box || box.y < 0 || box.y + box.height > 960)
    throw new Error("The split stamp is outside the capture");
  const timelineStarted = Date.now();
  await page.evaluate(async () => {
    const dial = document.querySelector<SVGElement>("#dial");
    const play = document.querySelector<HTMLButtonElement>("#hero #play");
    const stamp = document.querySelector<HTMLElement>("#stamp");
    if (!dial || !play || !stamp) throw new Error("Missing landing controls");
    const wait = (ms: number) =>
      new Promise((resolve) => setTimeout(resolve, ms));
    await wait(2_000);
    dial.dispatchEvent(
      new KeyboardEvent("keydown", { key: "ArrowLeft", bubbles: true }),
    );
    if (!stamp.hasAttribute("data-early"))
      throw new Error("The watch did not rewind before the split");
    const landed = new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        observer.disconnect();
        reject(new Error("The split did not arrive"));
      }, 10_000);
      const observer = new MutationObserver(() => {
        if (
          stamp.dataset.verdict === "separated" &&
          !stamp.hasAttribute("data-early")
        ) {
          play.click();
          observer.disconnect();
          clearTimeout(timeout);
          resolve();
        }
      });
      observer.observe(stamp, { attributes: true });
    });
    play.click();
    await landed;
    if (dial.getAttribute("aria-valuenow") !== "16")
      throw new Error("The split was not held at step 16");
    await wait(1_500);
    play.click();
    await wait(6_000);
  });
  const timelineSeconds = (Date.now() - timelineStarted) / 1000;
  const video = page.video();
  if (!video) throw new Error("Playwright did not start recording");
  await context.close();
  const source = await video.path();
  const duration = Number(
    execFileSync(
      "ffprobe",
      [
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        source,
      ],
      { encoding: "utf8" },
    ),
  );
  if (!Number.isFinite(duration) || duration < 10)
    throw new Error("The recorded video is too short");
  const offset = String(Math.max(0, duration - timelineSeconds));
  const palette = path.join(temporary, "palette.png");
  const gif = path.join(temporary, "demo.gif");
  ffmpeg(
    "-ss",
    offset,
    "-t",
    "10",
    "-i",
    source,
    "-vf",
    "fps=20,scale=960:-1:flags=lanczos,palettegen=max_colors=8:stats_mode=diff",
    palette,
  );
  ffmpeg(
    "-ss",
    offset,
    "-t",
    "10",
    "-i",
    source,
    "-i",
    palette,
    "-lavfi",
    "fps=20,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=none",
    "-loop",
    "0",
    gif,
  );
  if ((await stat(gif)).size >= 8_000_000)
    throw new Error("The GIF exceeds the 8 MB budget");
  await rename(gif, output);
  console.log(`Wrote ${output}`);
} finally {
  await browser?.close();
  await rm(temporary, { recursive: true, force: true });
}
