import { describe, it, expect, vi } from 'vitest';
import {
  runCleaningFlow,
  formatSummary,
  CleaningDeps,
  CleaningResult,
  MAX_PROPOSAL_CHARS,
} from '../../src/services/cleaningFlow';

function result(overrides: Partial<CleaningResult> = {}): CleaningResult {
  return {
    files: [
      {
        source_file: '/p/data/train.jsonl',
        relative_path: 'data/train.jsonl',
        output_path: '/p/.llm-training-agent/cleaned/data/train.jsonl',
        records_in: 10,
        records_out: 8,
        duplicates_removed: 2,
      },
    ],
    output_directory: '/p/.llm-training-agent/cleaned',
    total_files: 1,
    total_records_in: 10,
    total_records_out: 8,
    total_duplicates_removed: 2,
    records_preserved: true,
    llm_used: false,
    warnings: [],
    ...overrides,
  };
}

function makeDeps(answers: string[], overrides: Partial<CleaningDeps> = {}) {
  const said: string[] = [];
  const asked: string[] = [];
  const files: Record<string, string> = {
    '/p/data/train.jsonl': 'a\na\nb\n',
    '/p/.llm-training-agent/cleaned/data/train.jsonl': 'a\nb\n',
  };
  const deps = {
    getProjectRoot: vi.fn(async () => '/p'),
    ensureBackend: vi.fn(async () => true),
    ask: vi.fn(async (text: string) => {
      asked.push(text);
      return answers.shift() ?? '';
    }),
    say: vi.fn((text: string) => said.push(text)),
    withProgress: vi.fn(async (_title: string, task: () => Promise<unknown>) => task()),
    clean: vi.fn(async () => result()),
    readText: vi.fn(async (p: string) => files[p]),
    propose: vi.fn(async () => ({ changeId: 'c1' })),
    openChanges: vi.fn(async () => undefined),
    providerLabel: () => 'cohere / command-r',
    isLocalProvider: () => false,
    ...overrides,
  } as unknown as CleaningDeps & { clean: ReturnType<typeof vi.fn>; propose: ReturnType<typeof vi.fn> };
  return { deps, said, asked, files };
}

describe('runCleaningFlow', () => {
  it('uses scripts only (no AI) when the user picks scripts, then proposes a reviewable change', async () => {
    const { deps, said } = makeDeps(['scripts', 'yes', 'review']);
    await runCleaningFlow(deps);
    expect(deps.clean).toHaveBeenCalledWith({ projectPath: '/p', useLlm: false, removeDuplicates: true });
    expect(deps.propose).toHaveBeenCalledTimes(1);
    expect(deps.propose).toHaveBeenCalledWith('data/train.jsonl', 'a\nb\n', '/p', 'Dataset cleaning');
    expect(deps.openChanges).toHaveBeenCalledTimes(1);
    expect(said).toEqual([]);
  });

  it('only calls the AI when the user explicitly chooses it, and warns about cost first', async () => {
    const { deps, asked } = makeDeps(['llm', 'no', 'later']);
    await runCleaningFlow(deps);
    expect(deps.clean).toHaveBeenCalledWith({ projectPath: '/p', useLlm: true, removeDuplicates: false });
    expect(asked[0]).toContain('cohere / command-r');
    expect(asked[0]).toMatch(/credits|expensive/);
  });

  it('says a local model is free', async () => {
    const { deps, asked } = makeDeps(['cancel'], { isLocalProvider: () => true });
    await runCleaningFlow(deps);
    expect(asked[0]).toContain('free but can be slow');
  });

  it('skips the duplicates question when the user already asked for it', async () => {
    const { deps, asked } = makeDeps(['scripts', 'review']);
    await runCleaningFlow(deps, { removeDuplicates: true });
    expect(deps.clean).toHaveBeenCalledWith({ projectPath: '/p', useLlm: false, removeDuplicates: true });
    expect(asked.some((q) => q.includes('Also remove duplicate'))).toBe(false);
  });

  it('does nothing when the user cancels or closes the question', async () => {
    for (const answer of ['cancel', '']) {
      const { deps, said } = makeDeps([answer]);
      await runCleaningFlow(deps);
      expect(deps.clean).not.toHaveBeenCalled();
      expect(deps.propose).not.toHaveBeenCalled();
      expect(said[0]).toContain('did not change anything');
    }
  });

  it('never overwrites anything itself: it only proposes', async () => {
    const { deps } = makeDeps(['scripts', 'no', 'later']);
    await runCleaningFlow(deps);
    expect(Object.keys(deps)).not.toContain('writeText');
    expect(deps.propose).toHaveBeenCalled();
  });

  it('reports a failure instead of staying silent', async () => {
    const { deps, said } = makeDeps(['scripts', 'no'], {
      clean: vi.fn(async () => {
        throw new Error('backend exploded');
      }),
    });
    await runCleaningFlow(deps);
    expect(said[0]).toContain('Dataset cleaning failed: backend exploded');
  });

  it('tells the user when the backend is not running', async () => {
    const { deps, said } = makeDeps(['scripts', 'no'], { ensureBackend: vi.fn(async () => false) });
    await runCleaningFlow(deps);
    expect(deps.clean).not.toHaveBeenCalled();
    expect(said[0]).toContain('backend is not running');
  });

  it('explains when there is no dataset', async () => {
    const { deps, said } = makeDeps(['scripts', 'no'], {
      clean: vi.fn(async () => result({ files: [], total_files: 0 })),
    });
    await runCleaningFlow(deps);
    expect(said[0]).toContain('did not find any dataset files');
  });

  it('does not propose files that are already clean', async () => {
    const { deps, files, said } = makeDeps(['scripts', 'no']);
    files['/p/.llm-training-agent/cleaned/data/train.jsonl'] = files['/p/data/train.jsonl'];
    await runCleaningFlow(deps);
    expect(deps.propose).not.toHaveBeenCalled();
    expect(said[0]).toContain('Already clean');
    expect(said[0]).toContain('original files were not touched');
  });

  it('does not propose files too large for a reviewable change, and says where the copy is', async () => {
    const { deps, files, said } = makeDeps(['scripts', 'no']);
    files['/p/.llm-training-agent/cleaned/data/train.jsonl'] = 'x'.repeat(MAX_PROPOSAL_CHARS + 1);
    await runCleaningFlow(deps);
    expect(deps.propose).not.toHaveBeenCalled();
    expect(said[0]).toContain('Too large to review');
    expect(said[0]).toContain('/p/.llm-training-agent/cleaned');
  });

  it('never proposes a path that escapes the project', async () => {
    const { deps } = makeDeps(['scripts', 'no'], {
      clean: vi.fn(async () =>
        result({ files: [{ source_file: '/other/x.jsonl', relative_path: '../other/x.jsonl', output_path: '/p/c/x.jsonl', records_in: 1, records_out: 1 }] })
      ),
    });
    await runCleaningFlow(deps);
    expect(deps.propose).not.toHaveBeenCalled();
  });

  it('keeps going and reports when one proposal fails', async () => {
    const { deps, said } = makeDeps(['scripts', 'no'], {
      propose: vi.fn(async () => {
        throw new Error('too big for the backend');
      }),
    });
    await runCleaningFlow(deps);
    expect(said[0]).toContain('Could not prepare data/train.jsonl for review: too big for the backend');
  });
});

describe('formatSummary', () => {
  it('shows the record arithmetic and warns when it does not add up', () => {
    expect(formatSummary(result(), false)).toContain('10 in → 8 out (2 duplicates removed)');
    expect(formatSummary(result({ records_preserved: false }), false)).toContain('does not add up');
  });
});
