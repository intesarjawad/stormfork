export function element<T extends Element>(selector: string, type: abstract new () => T): T {
  const found = document.querySelector(selector);
  if (!(found instanceof type)) {
    throw new Error(`index.html is missing ${selector}`);
  }
  return found;
}

interface StateCard {
  readonly kind: "loading" | "error" | "empty";
  readonly title: string;
  readonly text: string;
  readonly command?: string;
  readonly detail?: string;
}

export function renderState(target: HTMLElement, card: StateCard): void {
  target.dataset["kind"] = card.kind;
  const title = document.createElement("p");
  title.className = "label";
  title.textContent = card.title;
  const text = document.createElement("p");
  text.textContent = card.text;
  target.replaceChildren(title, text);
  if (card.command !== undefined) {
    const pre = document.createElement("pre");
    pre.className = "cmd";
    const code = document.createElement("code");
    code.textContent = card.command;
    pre.append(code);
    target.append(pre);
  }
  if (card.detail !== undefined) {
    const details = document.createElement("details");
    const summaryLine = document.createElement("summary");
    summaryLine.textContent = "Technical detail";
    const body = document.createElement("p");
    body.textContent = card.detail;
    details.append(summaryLine, body);
    target.append(details);
  }
  target.hidden = false;
}

export function describe(error: unknown): string {
  return error instanceof Error ? error.message : "unknown error";
}

export function option(value: string, label: string): HTMLOptionElement {
  const item = document.createElement("option");
  item.value = value;
  item.textContent = label;
  return item;
}

export function svgNode<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attributes: Readonly<Record<string, string>>,
  text?: string,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [name, value] of Object.entries(attributes)) {
    node.setAttribute(name, value);
  }
  if (text !== undefined) {
    node.textContent = text;
  }
  return node;
}
