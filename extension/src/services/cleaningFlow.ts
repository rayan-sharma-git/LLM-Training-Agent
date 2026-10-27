/**
 * "Clean my dataset" as an action the extension performs.
 *
 * The flow, in order:
 *   1. ask HOW to clean:
 *        - scripts (free): whitespace and control characters, duplicate records,
 *          empty or meaningless records. The dataset files are edited directly.
 *        - the AI model: all of that, plus it reads every record and fixes typos
 *          and broken sentences or drops what makes no sense. The data is sent in
 *          small chunks, so a paid API uses credits; the user decides.
 *   2. for the AI model only, ask WHERE the result goes:
 *        - directly into the dataset files (cannot be undone), or
 *        - copies in `.llm-training-agent/cleaned`, to compare first. The user can
 *          then merge them (originals replaced, the folder removed) or discard them.
 *   3. run the backend cleaner (an AI run is a background job that is polled, so a
 *      slow model never causes a timeout) and report what changed.
 *
 * Nothing else is written: no manifest, no logs.
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
  junk_removed?: number;
  changed?: boolean;
  llm_used?: boolean;
}

export interface CleaningResult {
  files: CleaningFileSummary[];
  skipped_files?: string[];
  output_directory?: string | null;
  in_place?: boolean;
  total_files: number;
  total_records_in: number;
  total_records_out: number;
  total_duplicates_removed?: number;
  total_junk_removed?: number;
  total_files_changed?: number;
  records_preserved: boolean;
  llm_used: boolean;
  warnings?: string[];
}

export interface CleaningRequest {
  projectPath?: string;
  datasetPaths?: string[];
  chunkSize?: number;
  maxChunkChars?: number;
  outputDirectory?: string;
  maxFiles?: number;
  useLlm?: boolean;
  removeDuplicates?: boolean;
  inPlace?: boolean;
}

export interface CleaningProgress {
  file_index?: number;
  file_total?: number;
  file_name?: string;
  chunk_index?: number;
  chunk_total?: number;
}

export interface CleaningDeps {
  getProjectRoot(): Promise<string | undefined>;
  ensureBackend(): Promise<boolean>;
  /** Show a question with buttons; resolves to the chosen action, or '' for none. */
  ask(text: string, buttons: Array<{ label: string; action: string }>): Promise<string>;
  say(text: string): void;
  /** Runs `task` under a progress notification; `report` updates its text, `isCancelled` turns true on Cancel. */
  withProgress<T>(
    title: string,
    task: (report: (message: string) => void, isCancelled: () => boolean) => Promise<T>
  ): Promise<T>;
  clean(options: CleaningRequest, onProgress: (p: CleaningProgress) => void, isCancelled: () => boolean): Promise<CleaningResult>;
  pendingCopies(projectRoot: string): Promise<string[]>;
  mergeCopies(projectRoot: string): Promise<string[]>;
  discardCopies(projectRoot: string): Promise<number>;
  /** e.g. "ollama / llama3.2" — shown so the user knows what would be called. */
  providerLabel(): string;
  isLocalProvider(): boolean;
}

export const COPIES_FOLDER = '.llm-training-agent/cleaned';

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;

export function describeProgress(p: CleaningProgress): string {
  const parts: string[] = [];
  if (p.file_total && p.file_total > 1 && p.file_index) {
    parts.push(`file ${p.file_index}/${p.file_total}`);
  }
  if (p.file_name) {
    parts.push(p.file_name);
  }
  if (p.chunk_total && p.chunk_index) {
    parts.push(`part ${p.chunk_index}/${p.chunk_total}`);
  }
  return parts.join(' · ');
}

export function formatSummary(result: CleaningResult, useLlm: boolean): string {
  const inPlace = result.in_place !== false;
  const duplicates = result.total_duplicates_removed ?? 0;
  const junk = result.total_junk_removed ?? 0;
  const removed = [
    duplicates ? plural(duplicates, 'duplicate') : '',
    junk ? `${junk} empty or meaningless` : '',
  ].filter(Boolean);

  const lines = [
    `✅ Cleaned ${plural(result.total_files, 'dataset file')} with ${useLlm && result.llm_used ? 'the AI model' : 'scripts'}.`,
    inPlace ? 'Your dataset files were updated directly.' : `Cleaned copies were saved in ${COPIES_FOLDER}; your originals are untouched.`,
    '',
    `- Records: ${result.total_records_in} in → ${result.total_records_out} out` + (removed.length ? ` (removed ${removed.join(', ')})` : ''),
  ];
  if (!result.records_preserved) {
    lines.push('- ⚠️ For some files the record count did not add up, so those files were left as they were.');
  }

  const changed = result.files.filter((f) => f.changed !== false && f.records_in !== f.records_out);
  for (const file of changed.slice(0, 8)) {
    lines.push(`- ${file.relative_path}: ${file.records_in} → ${file.records_out}`);
  }
  if (changed.length > 8) {
    lines.push(`- …and ${changed.length - 8} more`);
  }
  if (!changed.length && !(result.total_files_changed ?? 0)) {
    lines.push('- Nothing needed to be removed or fixed.');
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
      '• Scripts — free and fast, nothing leaves your computer. Fixes whitespace and control characters and removes duplicate, empty and meaningless records. Your dataset files are updated directly.\n' +
      '• AI model — does all of that and also reads every record: it can correct typos and broken sentences and remove what makes no sense. ' +
      providerNote,
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
  const useLlm = mode === 'llm';

  let inPlace = true;
  if (useLlm) {
    const where = await deps.ask(
      'Where should I put the AI-cleaned data?\n\n' +
        '• Separate folder (safer) — I save cleaned copies in ' + COPIES_FOLDER + ' so you can compare. ' +
        'If you like them, I merge them into your dataset and remove that folder.\n' +
        '• Change my files directly — your dataset files are overwritten right away and this cannot be undone.',
      [
        { label: 'Separate folder (safer)', action: 'copy' },
        { label: 'Change my files directly', action: 'inplace' },
        { label: 'Cancel', action: 'cancel' },
      ]
    );
    if (where !== 'copy' && where !== 'inplace') {
      deps.say('Okay, I did not change anything.');
      return;
    }
    inPlace = where === 'inplace';
  }

  if (!(await deps.ensureBackend())) {
    deps.say('⚠️ The Python backend is not running, so I could not clean the dataset. Check the "LLM Training Agent: Backend" output channel.');
    return;
  }

  let cancelled = false;
  let result: CleaningResult;
  try {
    result = await deps.withProgress(useLlm ? 'Cleaning dataset with the AI model…' : 'Cleaning dataset with scripts…', (report, isCancelled) =>
      deps.clean(
        { projectPath: root, useLlm, removeDuplicates: request.removeDuplicates !== false, inPlace },
        (progress) => report(describeProgress(progress)),
        () => {
          cancelled = isCancelled();
          return cancelled;
        }
      )
    );
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    if (cancelled) {
      deps.say(
        inPlace
          ? 'Cleaning stopped. Files that were already finished have been updated; the rest are unchanged.'
          : 'Cleaning stopped. Your original files are unchanged.'
      );
    } else {
      deps.say(`⚠️ Dataset cleaning failed: ${reason}`);
    }
    return;
  }

  if (!result.total_files) {
    deps.say(
      'I did not find any dataset files (.jsonl, .json, .csv, …) in this project. ' +
        'Put your dataset inside the open folder and try again.'
    );
    return;
  }

  const summary = formatSummary(result, useLlm);
  const copied = result.files.filter((f) => f.output_path).length;
  if (inPlace || !copied) {
    deps.say(summary);
    return;
  }

  await offerMerge(deps, root, `${summary}\n\nWhat should I do with the cleaned copies?`);
}

/** Ask whether to merge or discard the cleaned copies, then do it. */
async function offerMerge(deps: CleaningDeps, root: string, question: string): Promise<void> {
  const choice = await deps.ask(question, [
    { label: 'Merge into my dataset', action: 'merge' },
    { label: 'Keep them for now', action: 'later' },
    { label: 'Discard them', action: 'discard' },
  ]);
  if (choice === 'merge') {
    try {
      const merged = await deps.mergeCopies(root);
      deps.say(
        `✅ Merged ${plural(merged.length, 'cleaned file')} into your dataset: ${merged.join(', ') || 'nothing to merge'}. ` +
          `The ${COPIES_FOLDER} folder was removed.`
      );
    } catch (error) {
      deps.say(`⚠️ Could not merge the cleaned copies: ${error instanceof Error ? error.message : String(error)}`);
    }
  } else if (choice === 'discard') {
    try {
      const count = await deps.discardCopies(root);
      deps.say(`Discarded ${plural(count, 'cleaned copy')}. Your original files were not touched.`);
    } catch (error) {
      deps.say(`⚠️ Could not discard the cleaned copies: ${error instanceof Error ? error.message : String(error)}`);
    }
  } else {
    deps.say(`Okay. The cleaned copies stay in ${COPIES_FOLDER}; your originals are unchanged. Tell me "merge the cleaned dataset" whenever you want to use them.`);
  }
}

/** "Merge the cleaned dataset" / "discard the cleaned copies": act on copies saved earlier. */
export async function runMergeFlow(deps: CleaningDeps): Promise<void> {
  const root = await deps.getProjectRoot();
  if (!root) {
    return;
  }
  if (!(await deps.ensureBackend())) {
    deps.say('⚠️ The Python backend is not running. Check the "LLM Training Agent: Backend" output channel.');
    return;
  }
  let pending: string[];
  try {
    pending = await deps.pendingCopies(root);
  } catch (error) {
    deps.say(`⚠️ Could not look for cleaned copies: ${error instanceof Error ? error.message : String(error)}`);
    return;
  }
  if (!pending.length) {
    deps.say('There are no cleaned copies waiting. Ask me to clean your dataset with the AI model and choose the separate folder to get some.');
    return;
  }
  await offerMerge(
    deps,
    root,
    `${plural(pending.length, 'cleaned file')} waiting in ${COPIES_FOLDER}: ${pending.slice(0, 8).join(', ')}` +
      `${pending.length > 8 ? ', …' : ''}.\n\nMerging replaces your original dataset files with them and removes that folder.`
  );
}
