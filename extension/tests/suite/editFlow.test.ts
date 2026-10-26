import { describe, it, expect, vi } from 'vitest';
import { runEditFlow, EditDeps, EditSuggestion } from '../../src/services/editFlow';

function makeDeps(answers: string[], overrides: Partial<EditDeps> = {}, suggestion: Partial<EditSuggestion> = {}) {
  const said: string[] = [];
  const asked: string[] = [];
  const deps = {
    getProjectRoot: vi.fn(async () => '/p'),
    ensureBackend: vi.fn(async () => true),
    findFile: vi.fn(async () => ['train.py']),
    suggest: vi.fn(async () => ({ proposedContent: 'lr = 1e-4\n', explanation: 'Lowered lr.', ...suggestion })),
    propose: vi.fn(async () => ({})),
    ask: vi.fn(async (text: string) => {
      asked.push(text);
      return answers.shift() ?? '';
    }),
    say: vi.fn((text: string) => said.push(text)),
    withProgress: vi.fn(async (_t: string, task: () => Promise<unknown>) => task()),
    openChanges: vi.fn(async () => {}),
    providerLabel: vi.fn(() => 'ollama / llama3.2'),
    ...overrides,
  } as unknown as EditDeps & Record<string, ReturnType<typeof vi.fn>>;
  return { deps, said, asked };
}

const request = { file: 'train.py', instruction: 'set the learning rate to 1e-4 in train.py' };

describe('runEditFlow', () => {
  it('proposes the change for review and never writes the file itself', async () => {
    const { deps } = makeDeps(['review']);
    await runEditFlow(deps, request);
    expect(deps.suggest).toHaveBeenCalledWith('train.py', request.instruction, '/p');
    expect(deps.propose).toHaveBeenCalledWith('train.py', 'lr = 1e-4\n', '/p', request.instruction);
    expect(deps.openChanges).toHaveBeenCalled();
  });

  it('says nothing was applied and where to find the change when the user is not ready', async () => {
    const { deps, said } = makeDeps(['later']);
    await runEditFlow(deps, request);
    expect(deps.openChanges).not.toHaveBeenCalled();
    expect(said.join('\n')).toContain('View Changes');
  });

  it('stops when the file cannot be found', async () => {
    const { deps, said } = makeDeps([], { findFile: vi.fn(async () => []) } as Partial<EditDeps>);
    await runEditFlow(deps, request);
    expect(deps.suggest).not.toHaveBeenCalled();
    expect(said[0]).toContain('could not find');
  });

  it('asks the user to be specific when several files match, and changes nothing', async () => {
    const { deps, said } = makeDeps([], { findFile: vi.fn(async () => ['a/train.py', 'b/train.py']) } as Partial<EditDeps>);
    await runEditFlow(deps, request);
    expect(deps.suggest).not.toHaveBeenCalled();
    expect(said[0]).toContain('a/train.py');
    expect(said[0]).toContain('b/train.py');
  });

  it('reports a provider failure in plain words and proposes nothing', async () => {
    const { deps, said } = makeDeps([], {
      suggest: vi.fn(async () => {
        throw new Error('Ollama does not have the model "llama3.2".');
      }),
    } as Partial<EditDeps>);
    await runEditFlow(deps, request);
    expect(deps.propose).not.toHaveBeenCalled();
    expect(said.join('\n')).toContain('Ollama does not have the model');
  });

  it('does not propose when the model changed nothing', async () => {
    const { deps, said } = makeDeps([], {}, { unchanged: true });
    await runEditFlow(deps, request);
    expect(deps.propose).not.toHaveBeenCalled();
    expect(said.join('\n')).toContain('did not change');
  });

  it('warns when the new version looks cut short', async () => {
    const { deps, asked } = makeDeps(['later'], {}, { suspectTruncation: true });
    await runEditFlow(deps, request);
    expect(asked[0]).toContain('much shorter');
  });

  it('does nothing without a backend', async () => {
    const { deps, said } = makeDeps([], { ensureBackend: vi.fn(async () => false) } as Partial<EditDeps>);
    await runEditFlow(deps, request);
    expect(deps.suggest).not.toHaveBeenCalled();
    expect(said.join('\n')).toContain('backend is not running');
  });
});
