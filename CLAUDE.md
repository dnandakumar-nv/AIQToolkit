# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NVIDIA NeMo Agent Toolkit (`nvidia-nat`) — an enterprise framework for building, deploying, and optimizing AI agents across multiple frameworks (LangChain, LlamaIndex, CrewAI, AutoGen, ADK, Semantic Kernel, Agno, Strands, etc.). Provides instrumentation, observability, evaluation, RL-based fine-tuning, and NVIDIA Dynamo integration.

## Terminology

- **Full name**: "NVIDIA NeMo Agent Toolkit" (first use, titles, headings)
- **Short name**: "NeMo Agent Toolkit" or "the toolkit" (subsequent references)
- **Code identifiers**: `nat` (API namespace/CLI), `nvidia-nat` (package name), `NAT_` (env vars)
- **Never** use deprecated names: "AgentIQ", "AIQ", "aiqtoolkit", "Agent Intelligence toolkit"

## Build & Development Commands

**Package manager**: `uv` with `setuptools` backend. Python 3.11–3.13.

```bash
# Install with specific integration
uv pip install "nvidia-nat[langchain]"
# Install everything
uv pip install "nvidia-nat[most]"

# Format (yapf) and lint (ruff) via pre-commit
pre-commit run --all-files

# Format only
yapf -i --style ./pyproject.toml <file>
# Lint only
ruff check --fix <file>
```

## Testing

```bash
# Run unit tests for a specific package (always specify package subdirectory)
pytest packages/nvidia_nat_core

# Run tests for multiple packages
pytest packages/nvidia_nat_core packages/nvidia_nat_langchain

# Include slow tests (>30s)
pytest --run_slow packages/nvidia_nat_core

# Include integration tests (require external services)
pytest --run_integration packages/nvidia_nat_core

# Run a single test file
pytest packages/nvidia_nat_core/tests/test_example.py

# Run a single test
pytest packages/nvidia_nat_core/tests/test_example.py::test_function_name
```

**Key test conventions:**
- Global 5-minute timeout per test (override with `@pytest.mark.timeout(seconds)`)
- `asyncio_mode = auto` — do NOT add `@pytest.mark.asyncio` to tests
- Markers: `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.benchmark`
- Fixtures use `@pytest.fixture(name="my_fixture")` with `fixture_` prefix on the function
- Use `nat_test_llm` (`_type: nat_test_llm`) for deterministic LLM stubbing without API keys
- Root `conftest.py` and `packages/nvidia_nat_test/` provide shared fixtures

## Code Style

- **Line length**: 120 characters
- **Formatter**: yapf (PEP 8 base), **Linter**: ruff (E, F, W, I, PL, UP rules)
- Force single-line imports, sorted by ruff isort
- All public APIs require Python 3.11+ type hints
- Google-style docstrings on public modules, classes, functions
- Every source file must start with the SPDX Apache-2.0 header (copy from existing file)
- Pydantic `SecretStr` fields: use `default=None` (optional) or `default_factory=lambda: SerializableSecretStr("")` (required), never `default=""`
- Exception handling: bare `raise` with `logger.error()` when re-raising; `logger.exception()` when not re-raising

## Monorepo Architecture

```
packages/                          # 30+ installable packages
├── nvidia_nat_core/               # Core: LLM providers, builder, context, profiler, eval, CLI
│   └── src/nat/                   # All core code under `nat` namespace
├── nvidia_nat_langchain/          # LangChain integration
├── nvidia_nat_llama_index/        # LlamaIndex integration
├── nvidia_nat_crewai/             # CrewAI integration
├── nvidia_nat_a2a/                # Agent-to-Agent protocol
├── nvidia_nat_mcp/                # Model Context Protocol
├── nvidia_nat_test/               # Shared test utilities and fixtures
├── nvidia_nat_eval/               # Evaluation system
└── ...                            # ~20 more integration packages
examples/                          # 50+ example workflows (each an installable package)
tests/                             # Root test suite
```

Each package has its own `pyproject.toml` declaring dependencies on `nvidia-nat` or `nvidia-nat-*` using `~=<version>` format. The root `pyproject.toml` maps all packages as editable `uv.sources`.

## Core Architecture (nvidia_nat_core)

### Component Registration System
Components are registered via Python entry points (`nat.components`, `nat.front_ends`, `nat.registry_handlers`, `nat.cli`). `ComponentEnum` defines types (LLM_CLIENT, MIDDLEWARE, FUNCTION, RETRIEVER_CLIENT, etc.) and `ComponentGroup` organizes them. The `Builder` class constructs workflows from YAML config by resolving component references.

### Context Propagation
`ContextState` (singleton in `builder/context.py`) manages request-scoped state via `ContextVar` for async safety:
- Request context: `conversation_id`, `user_message_id`, `workflow_run_id`
- Tracing: `workflow_trace_id`, `active_span_id_stack`
- Function tracking: `function_path_stack`
- Event streaming: reactive `Subject` for intermediate steps

Access via `Context.get()`. Scoped overrides via `Context.scope(**kwargs)`.

### Intermediate Steps & Observability
`IntermediateStep` models capture function calls, tool invocations, and LLM interactions (types: FUNCTION_START, FUNCTION_END, TOOL_CALL, LLM_RESPONSE). Streamed via reactive Subject for real-time UI. Integrates with OpenTelemetry spans.

### Workflow YAML Config
Workflows define agents as YAML with `_type` specifying the agent kind:
- `react_agent`: ReAct loop (Think → Act → Observe)
- `tool_calling_agent`: Schema-driven function calling
- `rewoo_agent`: Plan-then-execute (ReWOO)
- `reasoning_agent`: Wraps another agent with reasoning layer

### CLI
Entry point: `nat` command (registered via `nat.cli` entry points). Subcommands include `workflow`, `eval`, `optimize`.

## Adding Dependencies

New dependencies go in the relevant package's `pyproject.toml` (alphabetically sorted). Run `uv lock` for root and `uv lock --project packages/<pkg>` for individual packages. Pre-commit hooks verify lock files.

## CI Requirements

- All files need SPDX Apache-2.0 headers with current copyright year
- `pre-commit run --all-files` must pass
- Commits require `--signoff` (DCO compliance)
- Versions are derived by `setuptools-scm` — never hardcode
