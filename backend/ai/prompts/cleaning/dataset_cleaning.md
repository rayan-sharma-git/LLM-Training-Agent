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

## What to clean

1. Trim leading/trailing whitespace from every string value.
2. Normalise newlines (`\r\n` -> `\n`) and collapse runs of 3 or more blank lines to 2.
3. Remove non-printable control characters (keep `\n`, `\t`).
4. Fix obvious typos, doubled words and broken spacing inside prompt/response text.
5. Make punctuation and capitalisation of prompt/response text internally consistent.
6. Ensure every record keeps its original keys; keep the original wording and meaning.

## Hard rules

- Return **exactly the same number of records** you received.
- Return records in **exactly the same order**.
- Keep the **same keys** for every record. Do not add new keys.
- Do **not** merge, split, drop, deduplicate, reorder or invent records.
- Do **not** delete a record even if it looks empty or low quality; only flag it.
- Do **not** translate, summarise, shorten or extend the content.
- Do **not** change numeric or boolean values unless they are stored as broken text.
- If a value is already clean, return it unchanged.

## Output format

Return **only** a JSON object in this exact shape, with no commentary:

```json
{
  "records": [
    { "same_keys_as_input": "cleaned values" }
  ],
  "warnings": ["optional short notes about specific records"]
}
```

`records` must contain the cleaned chunk in the original order. `warnings` is optional.