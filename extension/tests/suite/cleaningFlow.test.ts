import { describe, it, expect, vi } from 'vitest';
import {
  runCleaningFlow,
  runMergeFlow,
  formatSummary,
  describeProgress,
  CleaningDeps,
  CleaningResult,
} from '../../src/services/cleaningFlow';

function result(overrides: Partial<CleaningResult> = {}): CleaningResult {
  return {
    files: [
      { source_file: '/p/data/train.jsonl', relative_path: 'data/train.jsonl', records_in: 10, records_out: 8, duplicates_removed: 2, changed: true },
    ],
    in_place: true,
    total_files: 1,
    total_records_in: 10,
    total_records_out: 8,
    total_duplicates_removed: 2,
    total_junk_removed: 0,
    total_files_changed: 1,
    records_preserved: true,
    llm_used: false,
    warnings: [],
    ...overrides,
  };
}

const copyResult = () =>
  result({
    in_place: false,
    llm_used: true,
    files: [
      {
        source_file: '/p/data/train.jsonl',
        relative_path: 'data/train.jsonl',
        output_path: '/p/.llm-training-agent/cleaned/data/train.jsonl',
        records_in: 10,
        records_out: 9,
        changed: true,
      },
    ],
  });

function makeDeps(answers: string[], overrides: Partial<CleaningDeps> = {}) {
  const said: string[] = [];
  const asked: string[] = [];
  const buttons: string[][] = [];
  const deps = {
    getProjectRoot: vi.fn(async () => '/p'),
    ensureBackend: vi.fn(async () => true),
    ask: vi.fn(async (text: string, b: Array<{ label: string; action: string }>) => {
      asked.push(text);
      buttons.push(b.map((x) => x.action));
      return answers.shift() ?? '';
    }),
    say: vi.fn((text: string) => said.push(text)),
    withProgress: vi.fn(async (_t: string, task: (r: (m: string) => void, c: () => boolean) => Promise<unknown>) =>
      task(() => undefined, () => false)
    ),
    clean: vi.fn(async () => result()),
    pendingCopies: vi.fn(async () => ['data/train.jsonl']),
    mergeCopies: vi.fn(async () => ['data/train.jsonl']),
    discardCopies: vi.fn(async () => 1),
    providerLabel: () => 'cohere / command-r',
    isLocalProvider: () => false,
    ...overrides,
  } as unknown as CleaningDeps & Record<string, ReturnType<typeof vi.fn>>;
  return { deps, said, asked, buttons };
}

describe('runCleaningFlow: scripts', () => {
  it('edits the dataset files directly, with no AI and no extra questions', async () => {
    const { deps, said, asked } = makeDeps(['scripts']);
    await runCleaningFlow(deps);
    expect(asked).toHaveLength(1); // only "how", never "where" or "duplicates?"
    expect(deps.clean).toHaveBeenCalledTimes(1);
    expect((deps.clean as any).mock.calls[0][0]).toEqual({ projectPath: '/p', useLlm: false, removeDuplicates: true, inPlace: true });
    expect(said[0]).toContain('updated directly');
    expect(said[0]).toContain('10 in → 8 out');
    expect(deps.mergeCopies).not.toHaveBeenCalled();
  });

  it('never creates a separate folder in scripts mode', async () => {
    const { deps, said } = makeDeps(['scripts']);
    await runCleaningFlow(deps);
    expect(said.join('\n')).not.toContain('.llm-training-agent');
  });
});

describe('runCleaningFlow: AI model', () => {
  it('warns about cost and asks where to put the result before calling the AI', async () => {
    const { deps, asked, buttons } = makeDeps(['llm', 'cancel']);
    await runCleaningFlow(deps);
    expect(asked[0]).toContain('cohere / command-r');
    expect(asked[0]).toMatch(/credits|expensive/);
    expect(asked[1]).toContain('Where should I put');
    expect(buttons[1]).toEqual(['copy', 'inplace', 'cancel']);
    expect(deps.clean).not.toHaveBeenCalled();
  });

  it('says a local model is free', async () => {
    const { deps, asked } = makeDeps(['cancel'], { isLocalProvider: () => true });
    await runCleaningFlow(deps);
    expect(asked[0]).toContain('free but can be slow');
  });

  it('can change the files directly when the user chooses that', async () => {
    const { deps, said } = makeDeps(['llm', 'inplace'], { clean: vi.fn(async () => result({ llm_used: true })) });
    await runCleaningFlow(deps);
    expect((deps.clean as any).mock.calls[0][0]).toMatchObject({ useLlm: true, inPlace: true });
    expect(said[0]).toContain('the AI model');
    expect(deps.mergeCopies).not.toHaveBeenCalled();
  });

  it('saves copies, then merges them into the dataset on request', async () => {
    const { deps, said, buttons } = makeDeps(['llm', 'copy', 'merge'], { clean: vi.fn(async () => copyResult()) });
    await runCleaningFlow(deps);
    expect((deps.clean as any).mock.calls[0][0]).toMatchObject({ useLlm: true, inPlace: false });
    expect(buttons[2]).toEqual(['merge', 'later', 'discard']);
    expect(deps.mergeCopies).toHaveBeenCalledWith('/p');
    expect(said[0]).toContain('Merged 1 cleaned file');
    expect(said[0]).toContain('folder was removed');
  });

  it('can discard the copies, leaving the originals alone', async () => {
    const { deps, said } = makeDeps(['llm', 'copy', 'discard'], { clean: vi.fn(async () => copyResult()) });
    await runCleaningFlow(deps);
    expect(deps.discardCopies).toHaveBeenCalledWith('/p');
    expect(deps.mergeCopies).not.toHaveBeenCalled();
    expect(said[0]).toContain('original files were not touched');
  });

  it('keeps the copies for later and says how to merge them', async () => {
    const { deps, said } = makeDeps(['llm', 'copy', 'later'], { clean: vi.fn(async () => copyResult()) });
    await runCleaningFlow(deps);
    expect(deps.mergeCopies).not.toHaveBeenCalled();
    expect(deps.discardCopies).not.toHaveBeenCalled();
    expect(said[0]).toContain('merge the cleaned dataset');
  });

  it('shows progress while it runs', async () => {
    const reports: string[] = [];
    const { deps } = makeDeps(['llm', 'inplace'], {
      withProgress: vi.fn(async (_t: string, task: (r: (m: string) => void, c: () => boolean) => Promise<unknown>) =>
        task((m) => reports.push(m), () => false)
      ) as any,
      clean: vi.fn(async (_o: unknown, onProgress: (p: any) => void) => {
        onProgress({ file_index: 1, file_total: 2, file_name: 'train.jsonl', chunk_index: 3, chunk_total: 10 });
        return result();
      }) as any,
    });
    await runCleaningFlow(deps);
    expect(reports).toEqual(['file 1/2 · train.jsonl · part 3/10']);
  });
});

describe('runCleaningFlow: problems', () => {
  it('does nothing when the user cancels or closes the question', async () => {
    for (const answer of ['cancel', '']) {
      const { deps, said } = makeDeps([answer]);
      await runCleaningFlow(deps);
      expect(deps.clean).not.toHaveBeenCalled();
      expect(said[0]).toContain('did not change anything');
    }
  });

  it('reports a failure instead of staying silent', async () => {
    const { deps, said } = makeDeps(['scripts'], {
      clean: vi.fn(async () => {
        throw new Error('backend exploded');
      }),
    });
    await runCleaningFlow(deps);
    expect(said[0]).toContain('Dataset cleaning failed: backend exploded');
  });

  it('says what state the files are in when the user cancels a run', async () => {
    const { deps, said } = makeDeps(['llm', 'inplace'], {
      clean: vi.fn(async (_o: unknown, _p: unknown, isCancelled: () => boolean) => {
        isCancelled();
        throw new Error('Cleaning was cancelled.');
      }) as any,
      withProgress: vi.fn(async (_t: string, task: (r: (m: string) => void, c: () => boolean) => Promise<unknown>) =>
        task(() => undefined, () => true)
      ) as any,
    });
    await runCleaningFlow(deps);
    expect(said[0]).toContain('Cleaning stopped');
    expect(said[0]).toContain('already finished');
  });

  it('tells the user when the backend is not running', async () => {
    const { deps, said } = makeDeps(['scripts'], { ensureBackend: vi.fn(async () => false) });
    await runCleaningFlow(deps);
    expect(deps.clean).not.toHaveBeenCalled();
    expect(said[0]).toContain('backend is not running');
  });

  it('explains when there is no dataset', async () => {
    const { deps, said } = makeDeps(['scripts'], { clean: vi.fn(async () => result({ files: [], total_files: 0 })) });
    await runCleaningFlow(deps);
    expect(said[0]).toContain('did not find any dataset files');
  });

  it('warns when a file was left alone because its record count did not add up', async () => {
    const { deps, said } = makeDeps(['scripts'], { clean: vi.fn(async () => result({ records_preserved: false })) });
    await runCleaningFlow(deps);
    expect(said[0]).toContain('left as they were');
  });
});

describe('runMergeFlow', () => {
  it('lists the waiting copies and merges them', async () => {
    const { deps, said, asked } = makeDeps(['merge']);
    await runMergeFlow(deps);
    expect(asked[0]).toContain('data/train.jsonl');
    expect(deps.mergeCopies).toHaveBeenCalledWith('/p');
    expect(said[0]).toContain('Merged 1 cleaned file');
  });

  it('says so when there is nothing to merge', async () => {
    const { deps, said } = makeDeps([], { pendingCopies: vi.fn(async () => []) });
    await runMergeFlow(deps);
    expect(said[0]).toContain('no cleaned copies waiting');
    expect(deps.mergeCopies).not.toHaveBeenCalled();
  });

  it('reports a failed merge', async () => {
    const { deps, said } = makeDeps(['merge'], {
      mergeCopies: vi.fn(async () => {
        throw new Error('disk full');
      }),
    });
    await runMergeFlow(deps);
    expect(said[0]).toContain('Could not merge the cleaned copies: disk full');
  });
});

describe('formatSummary and describeProgress', () => {
  it('shows the record arithmetic', () => {
    expect(formatSummary(result({ total_junk_removed: 3 }), false)).toContain('10 in → 8 out (removed 2 duplicates, 3 empty or meaningless)');
  });
  it('says when nothing needed fixing', () => {
    const clean = result({ files: [{ source_file: '/p/a', relative_path: 'a.jsonl', records_in: 5, records_out: 5, changed: false }], total_files_changed: 0, total_duplicates_removed: 0 });
    expect(formatSummary(clean, false)).toContain('Nothing needed to be removed or fixed');
  });
  it('describes progress compactly', () => {
    expect(describeProgress({ file_name: 'a.jsonl', chunk_index: 1, chunk_total: 4 })).toBe('a.jsonl · part 1/4');
    expect(describeProgress({})).toBe('');
  });
});
