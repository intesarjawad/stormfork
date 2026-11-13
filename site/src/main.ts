import { element } from "./dom";
import { createAtlasRoute } from "./atlasRoute";
import { createScoreboardRoute } from "./scoreboardRoute";
import { fillConstants, openLanding, type Landing } from "./landing";
import { parseRoute, type Route } from "./route";
const DEFAULT_EXPERIMENT = "sandy_east_300km";

function main(): void {
  fillConstants();
  const summary = element("#summary", HTMLParagraphElement);
  const atlas = createAtlasRoute();
  const scores = createScoreboardRoute();
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let landing: Landing | null = null;
  let landingTicket = 0;
  function setRoute(name: string): void {
    document.documentElement.dataset["route"] = name;
    for (const link of document.querySelectorAll<HTMLAnchorElement>("[data-nav]")) {
      link.ariaCurrent = link.dataset["nav"] === name ? "page" : null;
    }
    window.scrollTo(0, 0);
  }

  function closeLanding(): void {
    landingTicket += 1;
    landing?.dispose();
    landing = null;
  }

  async function showLanding(card: boolean): Promise<void> {
    const ticket = ++landingTicket;
    const opened = await openLanding({ id: DEFAULT_EXPERIMENT, reducedMotion, card });
    if (ticket !== landingTicket) {
      opened.dispose();
      return;
    }
    landing = opened;
  }

  function applyRoute(route: Route): void {
    closeLanding();
    scores.close();
    if (route.kind === "experiment") {
      setRoute("atlas");
      if (atlas.id() === route.id) {
        atlas.applyView(route.view);
      } else {
        void atlas.open(route.id, route.view);
      }
      return;
    }
    atlas.close();
    if (route.kind === "how") {
      setRoute("how");
      summary.textContent = "How this site works.";
      return;
    }
    if (route.kind === "scores") {
      setRoute("scores");
      void scores.open(route.icao);
      return;
    }
    if (route.kind === "card") {
      setRoute("card");
      void showLanding(true);
      return;
    }
    const bare = ["", "#", "#/"].includes(window.location.hash);
    if (!bare) {
      setRoute("missing");
      summary.textContent = "Page not found.";
      return;
    }
    setRoute("home");
    summary.textContent = "The fork: Hurricane Sandy, starting position moved east.";
    void showLanding(false);
  }
  window.addEventListener("hashchange", () => {
    applyRoute(parseRoute(window.location.hash));
  });
  applyRoute(parseRoute(window.location.hash));
}

main();
