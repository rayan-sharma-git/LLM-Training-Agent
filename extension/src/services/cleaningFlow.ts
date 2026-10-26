/**
 * "Clean my dataset" as an action the extension performs.
 *
 * The flow, in order:
 *   1. ask HOW to clean — with plain scripts (free) or with the AI model
 *      (chunk by chunk; uses the user's provider and, if it is a paid API,
 *      their credits). The user decides; nothing is sent to an AI by default.
 *   2. ask whether to remove duplicate records (unless the user already said so);
 *   3. run the backend cleaner (cleaned COPIES go to .llm-training-agent/cleaned);
 *   4. propose each cleaned file as a change to the original, so the user
 *      reviews a diff and applies (or discards) it with "View Changes" — the
 *      original files are never overwritten silently, and every applied change
 *      can be rolled back.
 *
 * No `vscode` import: everything the flow needs comes in through `CleaningDeps`
 * so it can be tested end to end.
 */

export interface CleaningFileSummary {
  source_file: string;
  relative_path: string;
  output_path?: string | null;
  records_in: number;
  records_out: number;
  duplicates_removed?: number;
  llm_used?: boolean;
}

export interface CleaningResult {
  files: CleaningFileSummary[];
  skipped_files?: string[];
  output_directory?: string | null;
  total_files: number;
  total_records_in: number;
  total_records_out: number;
  total_duplicates_removed?: number;
  records_preserved: boolean;
  llm_used: boolean;
  warnings?: string[];
}

export interface CleaningOptions {
  projectPath: string;
  useLlm: boolean;
  removeDuplicates: boolean;
}

export interface CleaningDeps {
  getProjectRoot(): Promise<string | undefined>;
  ensureBackend(): Promise<boolean>;
  /** Show a question with buttons; resolves to the chosen action, or '' for none. */
  ask(text: string, buttons: Array<{ label: string; action: string }>): Promise<string>;
  say(text: string): void;
  withProgress<T>(title: string, task: () => Promise<T>): Promise<T>;
  clean(options: CleaningOptions): Promise<CleaningResult>;
  readText(absolutePath: string): Promise<string | undefined>;
  propose(relativePath: string, content: string, projectRoot: string, reason: string): Promise<unknown>;
  openChanges(): Promise<void>;
  /** e.g. "ollama / llama3.2" — shown so the user knows what would be called. */
  providerLabel(): string;
  isLocalProvider(): boolean;
}

/** The backend refuses proposals larger than this (see editing/file_editor.py). */
export const MAX_PROPOSAL_CHARS = 2_000_000;

export function formatSummary(result: CleaningResult, useLlm: boolean): string {
  const removed = result.total_duplicates_removed ?? 0;
  const lines = [
    `✅ Cleaned ${result.total_files} dataset file${result.total_files === 1 ? '' : 's'} ` +
      `with ${useLlm && result.llm_used ? 'the AI model' : 'scripts'}.`,
    '',
    `- Records: ${result.total_records_in} in → ${result.total_records_out} out` +
      (removed ? ` (${removed} duplicate${removed === 1 ? '' : 's'} removed)` : ''),
    `- Nothing lost: ${result.records_preserved ? 'every record is accounted for' : '⚠️ the record count does not add up — review before applying'}`,
  ];
  for (const file of result.files.slice(0, 8)) {
    const dup = file.duplicates_removed ? `, −${file.duplicates_removed} duplicates` : '';
    lines.push(`- ${file.relative_path}: ${file.records_in} → ${file.records_out}${dup}`);
  }
  if (result.files.length > 8) {
    lines.push(`- …and ${result.files.length - 8} more file(s)`);
  }
  for (const warning of (result.warnings ?? []).slice(0, 4)) {
    lines.push(`- ⚠️ ${warning}`);
  }
  return lines.join('\n');
}

export async function runCleaningFlow(
  deps: CleaningDeps,
  request: { removeDuplicates?: boolean } = {}
): Promise<void> {
  const root = await deps.getProjectRoot();
  if (!root) {
    return;
  }

  const providerNote = deps.isLocalProvider()
    ? `Your provider (${deps.providerLabel()}) runs on your own computer, so this is free but can be slow.`
    : `Your provider (${deps.providerLabel()}) is an online service. Every chunk of your dataset is sent to it, ` +
      'and with a paid API that uses your credits — on a big dataset this can be expensive.';

  const mode = await deps.ask(
    'How should I clean your dataset?\n\n' +
      '• Scripts — fast, free, nothing leaves your computer. Fixes whitespace and control characters and (if you want) removes duplicates.\n' +
      `• AI model — sends the data to the model in small chunks so it can also fix typos and formatting. ${providerNote}`,
    [
      { label: 'Use scripts (free)', action: 'scripts' },
      { label: 'Use the AI model', action: 'llm' },
      { label: 'Cancel', action: 'cancel' },
    ]
  );
  if (mode !== 'scripts' && mode !== 'llm') {
    deps.say('Okay, I did not change anything.');
    return;
  }

  let removeDuplicates = request.removeDuplicates;
  if (removeDuplicates === undefined) {
    const answer = await deps.ask('Also remove duplicate records (exact copies; the first one is kept)?', [
      { label: 'Yes, remove duplicates', action: 'yes' },
      { label: 'No, keep them', action: 'no' },
    ]);
    if (!answer) {
      deps.say('Okay, I did not change anything.');
      return;
    }
    removeDuplicates = answer === 'yes';
  }

  if (!(await deps.ensureBackend())) {
    deps.say('⚠️ The Python backend is not running, so I could not clean the dataset. Check the "LLM Training Agent: Backend" output channel.');
    return;
  }

  const useLlm = mode === 'llm';
  let result: CleaningResult;
  try {
    result = await deps.withProgress(useLlm ? 'Cleaning dataset with the AI model…' : 'Cleaning dataset with scripts…', () =>
      deps.clean({ projectPath: root, useLlm, removeDuplicates: Boolean(removeDuplicates) })
    );
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    deps.say(`⚠️ Dataset cleaning failed: ${reason}`);
    return;
  }

  if (!result.total_files) {
    deps.say(
      'I did not find any dataset files (.jsonl, .json, .csv, …) in this project. ' +
        'Put your dataset inside the open folder and try again.'
    );
    return;
  }

  let summary = formatSummary(result, useLlm);
  const proposed: string[] = [];
  const tooLarge: string[] = [];
  const unchanged: string[] = [];

  for (const file of result.files) {
    if (!file.output_path || !file.relative_path || /^[a-zA-Z]:|^\/|\.\./.test(file.relative_path)) {
      continue; // outside the project: leave it alone
    }
    const [cleaned, original] = await Promise.all([deps.readText(file.output_path), deps.readText(file.source_file)]);
    if (cleaned === undefined) {
      continue;
    }
    if (original !== undefined && cleaned.trim() === original.trim()) {
      unchanged.push(file.relative_path);
      continue;
    }
    if (cleaned.length > MAX_PROPOSAL_CHARS) {
      tooLarge.push(file.relative_path);
      continue;
    }
    try {
      await deps.propose(file.relative_path, cleaned, root, 'Dataset cleaning');
      proposed.push(file.relative_path);
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      summary += `\n- ⚠️ Could not prepare ${file.relative_path} for review: ${reason}`;
    }
  }

  if (unchanged.length) {
    summary += `\n- Already clean (no change): ${unchanged.join(', ')}`;
  }
  if (tooLarge.length) {
    summary +=
      `\n- Too large to review here: ${tooLarge.join(', ')}. The cleaned copies are in ` +
      `${result.output_directory ?? '.llm-training-agent/cleaned'} — copy them over the originals yourself if you are happy with them.`;
  }

  if (!proposed.length) {
    summary += `\n\nYour original files were not touched. Cleaned copies: ${result.output_directory ?? '.llm-training-agent/cleaned'}`;
    deps.say(summary);
    return;
  }

  const choice = await deps.ask(
    `${summary}\n\nI prepared ${proposed.length} change${proposed.length === 1 ? '' : 's'} to your dataset ` +
      `(${proposed.join(', ')}). Nothing has been overwritten yet — review the difference first.`,
    [
      { label: 'Review changes', action: 'review' },
      { label: 'Not now', action: 'later' },
    ]
  );
  if (choice === 'review') {
    await deps.openChanges();
  } else {
    deps.say('Your original files are unchanged. Run "LLM Training Agent: View Changes" whenever you want to review or apply the cleaned versions.');
  }
}
