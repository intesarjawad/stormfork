/** One contributor's rerun, as listed in an experiment's reproductions.json. */
export interface Reproduction {
  readonly by: string;
  readonly runtime: string;
  readonly passed: boolean;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Read the reproduction ledger, keeping only what the page counts.
 *
 * The ledger is validated in full when it is merged, so a malformed entry here
 * is an error rather than something to skip.
 */
export function parseReproductions(value: unknown): readonly Reproduction[] {
  if (!Array.isArray(value)) {
    throw new Error("reproductions.json is not a list");
  }
  return value.map((item: unknown) => {
    if (
      !isRecord(item) ||
      typeof item["by"] !== "string" ||
      typeof item["runtime"] !== "string" ||
      typeof item["passed"] !== "boolean"
    ) {
      throw new Error("reproductions.json contains a malformed entry");
    }
    return { by: item["by"], runtime: item["runtime"], passed: item["passed"] };
  });
}

function runtimes(count: number): string {
  return `${String(count)} runtime${count === 1 ? "" : "s"}`;
}

/**
 * The line shown under an experiment, or null before anyone has reproduced it.
 *
 * Only passing reruns count. People are distinct GitHub logins, which are not
 * case-sensitive; runtimes are distinct runtime descriptions.
 */
export function reproductionText(entries: readonly Reproduction[]): string | null {
  const passing = entries.filter((entry) => entry.passed);
  if (passing.length === 0) {
    return null;
  }
  const people = new Set(passing.map((entry) => entry.by.toLowerCase())).size;
  const kinds = new Set(passing.map((entry) => entry.runtime)).size;
  return `Reproduced by ${people === 1 ? "1 person" : `${String(people)} people`} on ${runtimes(kinds)}`;
}
