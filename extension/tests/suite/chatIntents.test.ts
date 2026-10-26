import { describe, it, expect } from 'vitest';
import { detectIntent } from '../../src/services/chatIntents';

describe('detectIntent: requests to act', () => {
  const cases: Array<[string, boolean]> = [
    ['remove duplicates from my dataset', true],
    ['Remove duplicates', true],
    ['please remove the duplicate records in train.jsonl', true],
    ['can you remove duplicates from my data?', true],
    ['dedupe the dataset', true],
    ['deduplicate my training data', true],
    ['clean my dataset', false],
    ['Clean up the dataset please', false],
    ['clean the data and remove duplicates', true],
    ['I want you to clean my training data', false],
    ['get rid of duplicates in the jsonl', true],
  ];
  for (const [text, dup] of cases) {
    it(`"${text}" -> clean (removeDuplicates=${dup})`, () => {
      expect(detectIntent(text)).toEqual({ kind: 'clean', removeDuplicates: dup });
    });
  }

  it('recognises an analyze-project request', () => {
    expect(detectIntent('analyze my project')).toEqual({ kind: 'analyze' });
    expect(detectIntent('please analyze this project')).toEqual({ kind: 'analyze' });
  });
});

describe('detectIntent: questions and chat go to the AI', () => {
  for (const text of [
    'how do I remove duplicates from a dataset?',
    'what does cleaning a dataset do',
    'why are there duplicates in my data',
    'should I clean my dataset before fine-tuning?',
    'explain deduplication',
    'is my dataset clean?',
    'what learning rate should I use',
    'hello',
    'fix my learning rate',
    '',
  ]) {
    it(`"${text}" -> not an action`, () => {
      expect(detectIntent(text)).toBeNull();
    });
  }

  it('ignores very long messages (pasted content)', () => {
    expect(detectIntent('clean my dataset ' + 'x'.repeat(400))).toBeNull();
  });
});

describe('detectIntent: changing a named file', () => {
  const edits: Array<[string, string]> = [
    ['set the learning rate to 1e-4 in train.py', 'train.py'],
    ['please change num_train_epochs to 5 in configs/train.yaml', 'configs/train.yaml'],
    ['add a comment explaining the loss to src\\losses.py', 'src/losses.py'],
    ['Increase the batch size in `train.py`', 'train.py'],
    ['update the model name in config.json', 'config.json'],
    ['can you reduce max_seq_length in train.py?', 'train.py'],
  ];
  for (const [text, file] of edits) {
    it(`"${text}" -> edit ${file}`, () => {
      const intent = detectIntent(text);
      expect(intent?.kind).toBe('edit');
      expect(intent && intent.kind === 'edit' ? intent.file : '').toBe(file);
    });
  }

  it('keeps the whole sentence as the instruction', () => {
    expect(detectIntent('set the learning rate to 1e-4 in train.py')).toEqual({
      kind: 'edit',
      file: 'train.py',
      instruction: 'set the learning rate to 1e-4 in train.py',
    });
  });

  it('datasets and duplicates still go to the cleaning flow, not the editor', () => {
    expect(detectIntent('remove duplicates from train.jsonl')).toEqual({ kind: 'clean', removeDuplicates: true });
    expect(detectIntent('delete duplicates')).toEqual({ kind: 'clean', removeDuplicates: true });
    expect(detectIntent('clean data.csv')).toEqual({ kind: 'clean', removeDuplicates: false });
  });

  for (const text of [
    'what is in train.py',
    'how do I change the learning rate in train.py?',
    'explain train.py',
    'train.py',
    'is the learning rate in train.py too high',
  ]) {
    it(`"${text}" -> a question, not an edit`, () => {
      expect(detectIntent(text)).toBeNull();
    });
  }
});
