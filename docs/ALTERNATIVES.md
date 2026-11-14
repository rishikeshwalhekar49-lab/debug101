# Rejected Alternatives

## System Choice: Orchestration Framework

### Rejected Option 1: Celery Task Queue
- Description: Distribute agent tasks as Celery workers with Redis broker.
- Why Rejected: Requires Redis infrastructure; debugging async task graphs is harder than explicit LangGraph edges; overkill for a single-machine CLI tool.
- What We Gave Up: Horizontal scalability across machines; persistent task retry history.

### Rejected Option 2: Plain Python `asyncio`
- Description: Coordinate agents with `async/await` coroutines and queues.
- Why Rejected: No built-in state machine; control flow logic would be scattered across coroutines; no ReAct loop support without reimplementing it.
- What We Gave Up: Zero framework dependency; lower startup overhead.

### Revisit Criteria:
- Reconsider if LangGraph introduces breaking changes in two consecutive major versions, or if the agent graph grows beyond 20 nodes and graph compilation time exceeds 10 s.

---

## System Choice: Vector Store for Experience Memory

### Rejected Option 1: PostgreSQL + pgvector
- Description: Single relational DB with native vector similarity via pgvector extension.
- Why Rejected: Requires a running Postgres server; setup friction for a CLI tool used by individual researchers; pgvector ANN index quality lags behind dedicated vector DBs.
- What We Gave Up: ACID transactions spanning both scalar and vector data; SQL query flexibility.

### Rejected Option 2: FAISS (in-memory flat index)
- Description: Facebook's pure-C++ vector search library with Python bindings.
- Why Rejected: No persistence without custom serialization; no metadata filtering; requires manual index management.
- What We Gave Up: Fastest possible ANN search (sub-ms); zero infrastructure.

### Revisit Criteria:
- Reconsider if the experience store needs to be shared across machines (requires a server-based DB), or if ChromaDB introduces a license change that restricts embedding use.

---

## System Choice: LLM Provider

### Rejected Option 1: OpenAI GPT-4o
- Description: Use OpenAI as the primary LLM provider.
- Why Rejected: Higher cost per token; rate limits on the free/low tier; Groq offers dramatically lower latency for inference-heavy workloads.
- What We Gave Up: GPT-4o's broader general knowledge and stronger code reasoning on some benchmarks.

### Rejected Option 2: Local `ollama` model (e.g., Llama 3.1)
- Description: Run a local open-weight model for all LLM calls.
- Why Rejected: Quality of local models on code generation and causal reasoning tasks is insufficient for production repair rates; GPU VRAM requirement conflicts with running the user's training script simultaneously.
- What We Gave Up: Zero API cost; full offline operation; no data sent to external providers.

### Revisit Criteria:
- Reconsider local models when 7B+ quantized models reliably pass the benchmark repair tests (`evaluate_pipeline.py`), or when an organization requires data sovereignty.

---

## System Choice: Validation Strategy

### Rejected Option 1: Unit Test Re-run Only
- Description: Validate patches by running the project's existing test suite.
- Why Rejected: Most ML training scripts have no unit tests; this would make the validator a no-op for the primary use case.
- What We Gave Up: Integration with the project's own quality gate.

### Rejected Option 2: Loss-Curve Comparison Only (No Multi-Seed)
- Description: Compare training loss before and after patch across a single run.
- Why Rejected: Single runs have high variance; a "fix" that gets lucky on one seed would be accepted; statistical significance requires multiple seeds.
- What We Gave Up: Speed — single-seed validation is ~5× faster.

### Revisit Criteria:
- Reconsider if multi-seed validation compute cost (5 seeds × training time) makes the tool impractical for large models. Add a `--fast` flag to skip Gate 2 for time-budget-constrained runs.
