# Incident Response Assistant – Evaluation, Profiling, and Optimization Demo

This end-to-end example shows how to use the NeMo Agent Toolkit (NAT) **evaluation**, **profiler**, and **optimizer** together on a realistic security operations workflow. The workflow assists a Security Operations Center (SOC) engineer in triaging high-severity incidents by:

- Searching a curated catalogue of internal runbooks with the `incident_runbook_search` tool.
- Guiding a ReAct agent to choose the best-matching runbook and produce a structured JSON response.
- Capturing detailed metrics about LLM usage, concurrency, and prompt efficiency with the profiler.
- Iteratively tuning prompt instructions and LLM hyperparameters using the optimizer to maximize evaluation scores while reducing latency and token consumption.

The repository includes:

| Artifact | Purpose |
| --- | --- |
| [`data/runbooks.json`](data/runbooks.json) | Scenario-rich incident response runbooks ranked by a custom scoring function. |
| [`data/eval_dataset.json`](data/eval_dataset.json) | Evaluation set containing realistic incident escalations, expected answers, and reference contexts for RAG metrics. |
| [`src/nat_incident_response_assistant/register.py`](src/nat_incident_response_assistant/register.py) | Registers the runbook search tool built on a lightweight lexical + heuristic scorer. |
| [`src/nat_incident_response_assistant/scoring.py`](src/nat_incident_response_assistant/scoring.py) | Tokenization and scoring utilities with interpretable evidence exposed to the agent. |
| [`src/nat_incident_response_assistant/prompts.py`](src/nat_incident_response_assistant/prompts.py) | Default prompt instructions encouraging grounded, tool-aware reasoning. |
| [`configs/eval_config.yml`](configs/eval_config.yml) | Baseline configuration for running `nat eval` with profiler integrations. |
| [`configs/optimizer_config.yml`](configs/optimizer_config.yml) | Configuration enabling both numeric and prompt optimization via `nat optimize`. |

The workflow relies only on NVIDIA NIM endpoints (set `NVIDIA_API_KEY`) and local JSON data—no other external services are required.

## 1. Environment setup

```bash
# From the repository root
uv pip install -e examples/evaluation_and_profiling/incident_response_assistant
```

This editable install exposes the `incident_runbook_search` tool and ensures NAT loads the package during evaluation or optimization runs.

Export your NVIDIA API key so the NIM-backed LLMs and embedders can be reached:

```bash
export NVIDIA_API_KEY=<your key>
```

## 2. Explore the workflow components

1. **Runbook search tool** – parses [`runbooks.json`](data/runbooks.json) into rich `Runbook` objects and ranks them using cosine similarity with heuristic boosts for matching tags, detection signals, and severity indicators. The tool returns structured JSON with scoring evidence so the agent can justify its decisions.
2. **Incident response agent** – implemented using NAT’s `react_agent`. It receives additional instructions from [`prompts.py`](src/nat_incident_response_assistant/prompts.py) that force structured JSON output (`recommended_runbook_id`, `action_plan`, etc.) and require citing tool evidence before final answers.
3. **Evaluation dataset** – [`eval_dataset.json`](data/eval_dataset.json) pairs realistic SOC escalations with the expected playbook and textual ground truth, enabling RAG metrics (accuracy, relevance, groundedness) as well as trajectory tracing.

## 3. Run evaluation with profiler instrumentation

The [`configs/eval_config.yml`](configs/eval_config.yml) file enables a comprehensive profiler suite in addition to multiple evaluators. Launch an evaluation run with:

```bash
nat eval --config_file examples/evaluation_and_profiling/incident_response_assistant/configs/eval_config.yml
```

What you get after the run:

- **Evaluation metrics** (`rag_accuracy`, `rag_groundedness`, `rag_relevance`, trajectory analysis) under `./.tmp/nat/.../eval`.
- **Profiler outputs** including latency histograms, concurrency spike detection, prompt caching prefix suggestions, token uniqueness forecasts, and per-LLM usage breakdowns.
- **Workflow traces** you can visualize in observability tooling (Weave, LangSmith, etc.) if configured globally.

Suggested exploration exercises for notebooks or workshops:

- Plot profiler-estimated run time distributions versus observed latency to reason about concurrency tuning.
- Inspect `standardized_results_all.csv` to compare per-incident evaluation scores.
- Review the serialized tool outputs to understand how the agent leverages the ranking evidence.

## 4. Optimize prompts and hyperparameters

The [`configs/optimizer_config.yml`](configs/optimizer_config.yml) file extends the baseline configuration with:

- **Numeric search** over the agent LLM’s `temperature`, `top_p`, and `max_tokens`.
- **Prompt genetic algorithm** using NAT’s built-in `prompt_init` and `prompt_recombiner` helpers. The optimizer LLM (`optimizer_coach`) proposes new additional instructions that are evaluated on the same dataset.
- **Multi-objective scoring** to simultaneously maximize accuracy/groundedness and minimize latency plus token usage.

Kick off an optimization study with:

```bash
nat optimize --config_file examples/evaluation_and_profiling/incident_response_assistant/configs/optimizer_config.yml
```

Key artifacts written to `./.tmp/examples/evaluation_and_profiling/incident_response_assistant/optimizer/` include:

- `best_params.json` – winning combination of prompt text and numeric parameters.
- `trial_history.csv` – chronological log of metrics per trial for custom visualization.
- `prompt_population` directory – intermediate prompt candidates and their lineage (great for teaching GA concepts).

## 5. Notebook-friendly analysis ideas

The example is designed as a teaching scaffold. Consider the following notebook sections:

1. **Dataset walkthrough** – visualize runbook coverage and the heuristic scoring breakdown produced by the tool.
2. **Evaluation deep dive** – load the profiler CSVs to chart latency vs. token consumption per incident.
3. **Optimizer replay** – reconstruct Pareto fronts from `trial_history.csv` and compare baseline vs. tuned prompt instructions.
4. **What-if experiments** – adjust `functions.incident_runbook_search.top_k` or remove certain runbooks to demonstrate evaluation-driven regression detection.

## 6. Troubleshooting tips

- Ensure `nvidia-nat[langchain,profiling]` extras are installed; the `pyproject.toml` for this package wires them automatically when using `uv`.
- If you hit rate limits, lower `ga_parallel_evaluations` in the optimizer config or adjust the profiler’s concurrency spike threshold.
- All output directories live under `./.tmp` by default—delete them between demos to start fresh.

With these pieces, you have a self-contained workflow that showcases the complete NAT lifecycle: high-fidelity evaluation, deep profiler insights, and automated optimization.
