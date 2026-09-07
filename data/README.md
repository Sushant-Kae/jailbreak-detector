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
  `synthetic_obfuscation`

## Expanding the dataset

Append new rows with the same three columns. Keep `suspicious` examples
**abstract** (e.g. `[synthetic instruction override attempt] ...`) — never add
real working jailbreak payloads. Try to keep the two classes roughly balanced.
