# Models and APIs

Every LLM call goes through `silent_bug_agent/semantic/router.py`. It picks the
model for each task and falls over to the backup provider when the main one hits
its free limit or fails. No Anthropic or OpenAI key is used.

## Providers

| Provider | Role | Key in `.env` | Endpoint | Where to get a key |
|---|---|---|---|---|
| Groq | Main | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` | https://console.groq.com/keys |
| Cerebras | Backup | `CEREBRAS_API_KEY` | `https://api.cerebras.ai/v1` | https://cloud.cerebras.ai |

Both speak the OpenAI chat-completions protocol, so one client
(`OpenAICompatibleClient`) serves both. One key is enough; two give more
headroom. With no key, every agent uses its rule-based path and says so.

## Model for each task

| Agent | Task name | Main model | Backup model | Why |
|---|---|---|---|---|
| 2: semantic reviewer | `semantic_review` | Groq `openai/gpt-oss-120b` | Cerebras `gpt-oss-120b` | Largest production model on Groq, reasoning, strict JSON |
| 4: diagnosis | `diagnosis` | Groq `openai/gpt-oss-120b` | Cerebras `gpt-oss-120b` | Cause-and-effect reasoning |
| 5: fix synthesizer | `fix` | Groq `openai/gpt-oss-120b` | Cerebras `gpt-oss-120b` | Writes the patch |
| 7: meta-evolution | `meta` | Groq `openai/gpt-oss-120b` | Cerebras `gpt-oss-120b` | Writes lessons from past failures |
| 1: runtime anomaly consolidator | `perception` | Groq `openai/gpt-oss-20b` | Cerebras `qwen-3.8-27b` | Short summary, saves the daily limit |
| 3: validation analyzer | `validation` | Groq `openai/gpt-oss-20b` | Cerebras `qwen-3.8-27b` | Reads a few numbers |
| 6: experience memory | (embeddings) | `all-MiniLM-L6-v2` via ChromaDB | word overlap | Runs on the CPU, no key |
| Supervisor, 0, hooks, AST rules, statistics | none | no model | none | Deterministic |

Override any choice in `.env`, for example `SBA_MODEL_FIX=cerebras/gpt-oss-120b`.
`python -m silent_bug_agent doctor` prints the active choice for every task.

## Free limits (checked 3 November 2026)

| Model | Requests/min | Requests/day | Tokens/min | Tokens/day |
|---|---|---|---|---|
| Groq `openai/gpt-oss-120b`, `openai/gpt-oss-20b` | 30 | 1,000 | 8,000 | 200,000 |
| Cerebras `gpt-oss-120b`, `qwen-3.8-27b` | 5 | not stated | 30,000 | 1,000,000 |

Groq's page labels these as Developer plan base limits, so check your own limits
at https://console.groq.com/settings/limits. One `fix` run makes about 5 to 10
calls. Prompts are cut to about 12,000 characters of source so a call fits under
Groq's 8,000 tokens a minute.

## The model trained in this project

Besides the hosted LLMs there is one small model of our own, the trace classifier
(`silent_bug_agent/learned/`). It is not an LLM. It reads the numbers a 150-step
traced run produces (gradient norms, gradients left over from before an update,
output and input statistics, dead ReLUs, the loss curve, ...) and a few facts from
the script's syntax tree, and gives a probability for each of nine bugs and "clean".
It is a numpy network of 1,178 weights (a 16.5 KB file) that ships inside the
package, runs on the CPU in microseconds and needs no key or download. Since
version 4 it takes part in ranking causes by default (`--learned active`). How it
is trained and how well it does are in [learned_classifier.md](learned_classifier.md).

## Things to know

- The training script you scan is sent to the provider. Check its terms before
  using confidential code.
- The first run that stores a repair downloads the 79 MB embedding model to
  `~/.cache/chroma`. Without internet, memory search falls back to word overlap.
- Keys are only read from the environment or `.env`. They never appear in logs,
  error messages or reports.
