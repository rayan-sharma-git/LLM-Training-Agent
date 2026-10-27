# Dataset Cleaning Task (single chunk)

You are cleaning **one chunk** of a larger training dataset file. Other chunks of the
same file are cleaned in separate, independent requests. Never assume you can see the
rest of the file.

## Chunk under review

- Source file: `<<FILE_NAME>>`
- Chunk: `<<CHUNK_INDEX>>` of `<<CHUNK_TOTAL>>`
- Records in this chunk: `<<RECORD_COUNT>>`
- File format: `<<FORMAT>>`

## Records (JSON array)

```json
<<RECORDS_JSON>>
```

## What to do

Clean every record thoroughly, the way a careful editor would:

1. Trim whitespace, normalise newlines (`\r\n` -> `\n`), collapse runs of 3+ blank lines to 2, and remove control characters (keep `\n`, `\t`).
2. Fix typos, doubled words, broken spacing, wrong punctuation and capitalisation.
3. Fix grammar. If a sentence is garbled or does not make sense, rewrite it so it does, keeping the original intent.
4. Fix text-encoding damage (for example `â€™` -> `’`), stray HTML tags and leftover markup that is clearly noise.
5. Make the response actually answer its prompt where the intended meaning is clear; do not add new facts.
6. If a record is hopeless - it is empty, random characters, or its meaning cannot be recovered - add `"_remove": true` to it (keep its other fields as they were).

## Rules

- Return **exactly the same number of records** you received, in **exactly the same order**.
- Keep the **same keys** in every record. Do not add other keys (except `_remove`).
- Do **not** merge, split, reorder or invent records. Duplicates are removed by a separate script, so ignore them.
- Do **not** translate, summarise, shorten or pad the content.
- Do **not** change numbers or booleans unless they are stored as broken text.
- Never blank out a field. Only use `_remove` for a record that is truly unusable, and rarely.
- If a value is already clean, return it unchanged.

## Output format

Return **only** a JSON object in this exact shape, with no commentary:

```json
{
  "records": [
    { "same_keys_as_input": "cleaned values" }
  ],
  "warnings": ["optional short notes"]
}
```

`records` holds the cleaned chunk in the original order.
