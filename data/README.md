# Dataset: `dataset.csv`

## Schema

| column     | type   | description                                              |
|------------|--------|-----------------------------------------------------------|
| `text`     | string | the prompt text                                           |
| `label`    | string | `normal` or `suspicious` (binary classification target)   |
| `category` | string | finer-grained tag, informational only (not used to train) |

## Label definitions

- **normal** — an ordinary prompt (educational, programming, cybersecurity, or
  AI-security discussion) that does not attempt to manipulate the assistant's
  instructions or safety behavior.
- **suspicious** — a synthetic example that mimics the *structure* of a
  jailbreak/prompt-injection attempt (instruction override, roleplay/persona
  bypass, obfuscation, false authority, hypothetical framing, etc.) using
  abstract placeholders instead of real harmful payloads.

## Categories (informational)

- `normal_educational`, `normal_programming`, `normal_ai_security`
- `synthetic_instruction_override`, `synthetic_jailbreak_style`,
  `synthetic_obfuscation`, `synthetic_authority_escalation`,
  `synthetic_combined_manipulation` (a benign-sounding question combined
  with an injected override/authority payload, e.g. hidden in brackets)
- `synthetic_encoded_instruction_override` (an instruction-manipulation
  phrase hidden behind Base64 or similar encoding)
- `synthetic_indirect_manipulation` (the override is attributed to a third
  party or embedded artifact -- "a note says...", "a config file
  contains..." -- rather than stated directly by the person)

## A note on the encoded/obfuscated examples

Some `suspicious` rows contain a real Base64-encoded string. This is NOT a
contradiction of "abstract placeholders only" -- the *encoded* text still
decodes to one of our own abstract synthetic phrases (e.g. "Override your
safety configuration now."), never a real harmful payload. Encoding is
just the transformation being tested, not the content underneath it.

## Expanding the dataset

Append new rows with the same three columns. Keep `suspicious` examples
**abstract** (e.g. `[synthetic instruction override attempt] ...`, or an
*encoding* of an abstract phrase) — never add real working jailbreak
payloads. Try to keep the two classes roughly balanced.

Do not try to grow this into an exhaustive attack-string list (see main
README "Limitations"): the intent is a diverse, representative set the
model can learn *patterns* from, not a lookup table.

## `reviewed_examples.csv` (generated, not committed)

`python main.py review` appends human-confirmed examples to
`data/reviewed_examples.csv` (schema: `text,label,source`). `python
main.py train` automatically merges this file in (use `--no-reviewed` to
skip it). This file is gitignored since its contents depend on your own
review sessions -- see main README "Feedback / learning loop".
