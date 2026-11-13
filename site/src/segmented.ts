export interface Segments {
  /** Re-read the select's value after code changed it. */
  readonly sync: () => void;
}

/**
 * Replace a native select with a radio group of buttons, keeping the select as the
 * source of truth so the rest of the page keeps reading and writing `select.value`.
 */
export function segmented(select: HTMLSelectElement): Segments {
  const label = document.querySelector(`label[for="${select.id}"]`);
  const group = document.createElement("div");
  group.className = "segments";
  group.setAttribute("role", "radiogroup");
  if (label !== null) {
    label.id = `${select.id}-label`;
    label.removeAttribute("for");
    group.setAttribute("aria-labelledby", label.id);
  }
  select.hidden = true;
  select.after(group);

  const buttons = (): HTMLButtonElement[] => [...group.querySelectorAll("button")];

  const sync = (): void => {
    for (const button of buttons()) {
      const on = button.dataset["value"] === select.value;
      button.setAttribute("aria-checked", String(on));
      button.tabIndex = on ? 0 : -1;
    }
  };

  const choose = (value: string): void => {
    select.value = value;
    select.dispatchEvent(new Event("change", { bubbles: true }));
    sync();
  };

  const build = (): void => {
    group.replaceChildren(
      ...[...select.options].map((option) => {
        const button = document.createElement("button");
        button.type = "button";
        button.setAttribute("role", "radio");
        button.dataset["value"] = option.value;
        button.textContent = option.textContent;
        button.addEventListener("click", () => {
          choose(option.value);
        });
        return button;
      }),
    );
    sync();
  };

  group.addEventListener("keydown", (event) => {
    const step = ["ArrowRight", "ArrowDown"].includes(event.key)
      ? 1
      : ["ArrowLeft", "ArrowUp"].includes(event.key)
        ? -1
        : 0;
    if (step === 0) {
      return;
    }
    event.preventDefault();
    const all = buttons();
    const index = all.findIndex((button) => button.getAttribute("aria-checked") === "true");
    const next = all[(index + step + all.length) % all.length];
    if (next?.dataset["value"] !== undefined) {
      choose(next.dataset["value"]);
      next.focus();
    }
  });

  new MutationObserver(build).observe(select, { childList: true });
  build();
  return { sync };
}
