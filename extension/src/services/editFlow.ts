/**
 * "Change this file" as an action the extension performs.
 *
 *   1. find the named file in the open project;
 *   2. ask the backend to have the AI model rewrite it (the backend sends the
 *      WHOLE file; chat messages are cut short, which would damage it);
 *   3. record the result as a PROPOSAL, never as a write;
 *   4. the user reviews the diff in VS Code and applies it (or rolls it back)
 *      from "View Changes".
 *
 * No `vscode` import: everything comes in through `EditDeps`, so it is testable.
 */

export interface EditSuggestion {
  proposedContent: string;
  explanation?: string;
  unchanged?: boolean;
  suspectTruncation?: boolean;
}

export interface EditDeps {
  getProjectRoot(): Promise<string | undefined>;
  ensureBackend(): Promise<boolean>;
  /** Workspace-relative paths whose name matches `name`. */
  findFile(name: string): Promise<string[]>;
  suggest(relativePath: string, instruction: string, projectRoot: string): Promise<EditSuggestion>;
  propose(relativePath: string, content: string, projectRoot: string, reason: string): Promise<unknown>;
  ask(text: string, buttons: Array<{ label: string; action: string }>): Promise<string>;
  say(text: string): void;
  withProgress<T>(title: string, task: () => Promise<T>): Promise<T>;
  openChanges(): Promise<void>;
  providerLabel(): string;
}

export async function runEditFlow(deps: EditDeps, request: { file: string; instruction: string }): Promise<void> {
  const root = await deps.getProjectRoot();
  if (!root) {
    return;
  }

  const candidates = await deps.findFile(request.file);
  if (candidates.length === 0) {
    deps.say(`I could not find "${request.file}" in this project, so I changed nothing.`);
    return;
  }
  if (candidates.length > 1) {
    deps.say(
      `"${request.file}" matches ${candidates.length} files:\n` +
        candidates.slice(0, 8).map((c) => `- ${c}`).join('\n') +
        '\n\nTell me which one, for example by writing its folder too.'
    );
    return;
  }
  const relativePath = candidates[0];

  if (!(await deps.ensureBackend())) {
    deps.say('⚠️ The Python backend is not running, so I could not prepare the change. Check the "LLM Training Agent: Backend" output channel.');
    return;
  }

  deps.say(`Asking ${deps.providerLabel()} to change ${relativePath}. The whole file is sent to it. Nothing is written until you approve.`);

  let suggestion: EditSuggestion;
  try {
    suggestion = await deps.withProgress(`Preparing the change to ${relativePath}…`, () =>
      deps.suggest(relativePath, request.instruction, root)
    );
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    deps.say(`⚠️ I could not prepare that change: ${reason}`);
    return;
  }

  if (suggestion.unchanged) {
    deps.say(`The model did not change ${relativePath}. Try describing the change more specifically.`);
    return;
  }

  try {
    await deps.propose(relativePath, suggestion.proposedContent, root, request.instruction.slice(0, 200));
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    deps.say(`⚠️ I could not record the change for review: ${reason}`);
    return;
  }

  const warning = suggestion.suspectTruncation
    ? '\n\n⚠️ The new version is much shorter than the original. The model may have left parts out, so read the difference carefully.'
    : '';
  const choice = await deps.ask(
    `I prepared a change to ${relativePath}.${suggestion.explanation ? `\n\n${suggestion.explanation}` : ''}${warning}\n\n` +
      'Your file has not been changed yet. Review the difference first.',
    [
      { label: 'Review change', action: 'review' },
      { label: 'Not now', action: 'later' },
    ]
  );
  if (choice === 'review') {
    await deps.openChanges();
  } else {
    deps.say('Your file is unchanged. Run "LLM Training Agent: View Changes" whenever you want to review, apply or discard it.');
  }
}
