/**
 * Generic copy-button wiring, bundled by esbuild into assets/copy.js: any
 * `<button data-copy-target="id">` copies the text content of the element
 * with that id (the "reproduce this with an agent" prompts, Phase 12.2).
 * Clipboard fallback mirrors the demo page's CTA button: select the visible
 * text so the user can copy manually.
 */

/**
 * Measurement hook: CTA copies only. Deliberately a no-op — the counter
 * choice is the owner's (DECISIONS #69: repo-stats only until an analytics
 * stance is picked). No cookies, no PII, ever.
 */
function measure(_event: string): void {
  // intentionally empty
}

function selectText(element: HTMLElement): void {
  const range = document.createRange();
  range.selectNodeContents(element);
  const selection = window.getSelection();
  selection?.removeAllRanges();
  selection?.addRange(range);
}

function wireCopyButtons(): void {
  const buttons = document.querySelectorAll<HTMLButtonElement>('button[data-copy-target]');
  for (const button of buttons) {
    const targetId = button.dataset['copyTarget'];
    if (targetId === undefined) {
      continue;
    }
    const target = document.getElementById(targetId);
    if (target === null) {
      continue;
    }
    const idleLabel = button.textContent;
    button.addEventListener('click', () => {
      const done = (): void => {
        button.textContent = 'Copied!';
        setTimeout(() => {
          button.textContent = idleLabel;
        }, 1500);
        measure(`copy:${targetId}`);
      };
      navigator.clipboard.writeText(target.textContent ?? '').then(done, () => {
        selectText(target);
      });
    });
  }
}

wireCopyButtons();
