/**
 * Recognise requests in the chat that the extension should *do* rather than
 * just talk about ("remove duplicates from my dataset").
 *
 * Plain keyword rules on purpose: they work with every provider (small local
 * models cannot call tools reliably), cost nothing, and are easy to test. A
 * question such as "how do I remove duplicates?" is NOT an action and goes to
 * the AI as usual.
 */
export type ChatIntent =
  | { kind: 'clean'; removeDuplicates: boolean }
  | { kind: 'analyze' }
  /** Change one named file as instructed ("set the learning rate to 1e-4 in train.py"). */
  | { kind: 'edit'; file: string; instruction: string };

const POLITE_PREFIX =
  /^(?:please\s+|pls\s+|kindly\s+|ok(?:ay)?[,.]?\s+|hey[,.]?\s+|can you\s+|could you\s+|would you\s+|will you\s+|i want you to\s+|i need you to\s+|i'd like you to\s+|go ahead and\s+|just\s+)+/i;

const QUESTION_START =
  /^(?:how|what|why|which|when|where|who|is|are|does|do|did|should|explain|tell me|show me how|describe)\b/i;

const DATA_WORDS = /\b(?:data\s?sets?|data|records?|rows?|samples?|examples?|jsonl|csv|json|files?|training\s+set)\b/i;
const CLEAN_VERBS = /\b(?:clean(?:\s?up)?|de-?dup(?:licate)?s?|dedupe|(?:remove|delete|erase|eliminate|drop)\s+(?:the\s+|all\s+|any\s+)*(?:duplicat\w*|dupes?)|get\s+rid\s+of\s+(?:the\s+)?(?:duplicat\w*|dupes?)|normali[sz]e|saniti[sz]e|fix)\b/i;
const DUPLICATE_WORDS = /\b(?:de-?dup\w*|dedupe|duplicat\w*|dupes?)\b/i;
const ANALYZE_VERBS = /^(?:analy[sz]e|scan|inspect|review)\b/i;
const PROJECT_WORDS = /\b(?:project|repo|repository|codebase|workspace|training\s+setup|fine-?tun\w*)\b/i;

// Files the agent may be asked to change. Datasets are NOT here: they go
// through the cleaning flow, which works in chunks and keeps count of records.
const EDITABLE_FILE = /(?:^|[\s"'`(])((?:[\w.-]+[\\/])*[\w.-]+\.(?:py|ts|js|yaml|yml|toml|cfg|ini|sh|md|txt|json))(?=$|[\s"'`),:;])/i;
const EDIT_VERBS =
  /^(?:edit|change|modify|update|fix|add|rewrite|refactor|rename|set|increase|decrease|reduce|raise|lower|replace|remove|delete|make|use|switch|enable|disable|comment|optimi[sz]e|improve|adjust|tweak|bump|turn)\b/i;

export function detectIntent(rawText: string): ChatIntent | null {
  const text = (rawText || '').trim().replace(/[?.!]+$/, '');
  if (!text || text.length > 300) {
    return null;
  }
  const core = text.replace(POLITE_PREFIX, '').trim();
  if (!core || QUESTION_START.test(core)) {
    return null;
  }

  if (ANALYZE_VERBS.test(core) && PROJECT_WORDS.test(core) && !DATA_WORDS.test(core.replace(PROJECT_WORDS, ''))) {
    return { kind: 'analyze' };
  }

  // "set the learning rate to 1e-4 in train.py": a verb up front plus a file we can edit.
  if (!DUPLICATE_WORDS.test(core)) {
    const file = EDITABLE_FILE.exec(core);
    if (file && EDIT_VERBS.test(core)) {
      return { kind: 'edit', file: file[1].replace(/\\/g, '/'), instruction: core };
    }
  }

  const wantsDuplicates = DUPLICATE_WORDS.test(core);
  const startsWithAction = CLEAN_VERBS.test(core.split(/\s+/).slice(0, 4).join(' '));
  if (startsWithAction && (DATA_WORDS.test(core) || wantsDuplicates)) {
    return { kind: 'clean', removeDuplicates: wantsDuplicates };
  }
  return null;
}
