<!--
Copyright (c) 2025-2026, NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
<!-- path-check-skip-file -->

> [!NOTE]
> **EXPERIMENTAL**: This integration between NeMo Agent Toolkit and Dynamo is experimental and under active development. APIs, configurations, and features may change without notice.

# Complex LangGraph Benchmark Agent

**Complexity:** Advanced

A 12-node LangGraph-based IT Incident Response agent designed to stress-test NVIDIA Dynamo inference optimizations (KV cache prefix routing, Thompson Sampling, worker stickiness). The agent generates **9-18 LLM calls per query** through parallel branches, conditional edges, and retry cycles, making it ideal for benchmarking multi-call LLM workflows at scale.

> [!IMPORTANT]
> **Prerequisite**: Before running evaluations against Dynamo, complete the [Dynamo Backend Setup Guide](../../../external/dynamo/README.md) to set up and verify your Dynamo inference server.

## Table of Contents

1. [Agent Overview](#agent-overview)
2. [Graph Architecture](#graph-architecture)
3. [Graph Nodes](#graph-nodes)
4. [Conditional Edges](#conditional-edges)
5. [Incident Tools](#incident-tools)
6. [State Schema](#state-schema)
7. [Running Evaluations](#running-evaluations)
8. [Configuration Reference](#configuration-reference)
9. [Dataset](#dataset)
10. [Troubleshooting](#troubleshooting)

---

## Agent Overview

The agent processes IT incident reports through a multi-stage pipeline:

1. **Classifies** the incident by severity and category
2. **Analyzes** the incident across 3 parallel branches (security, performance, infrastructure)
3. **Assesses quality** of the analysis with a retry loop
4. **Synthesizes** a report with a critique/revision cycle
5. **Extracts** recommended tool-call actions
6. **Assesses risk** level
7. **Generates** a comprehensive final summary

Key architectural features:
- **Parallel fan-out/fan-in**: 3 concurrent `analyze_branch` nodes via LangGraph `Send()`
- **Quality gate cycle**: Re-dispatches parallel analysis if quality score falls below threshold
- **Critique cycle**: Loops between `synthesize_report` and `critique_report` until report is accepted
- **Decision-only tools**: 6 incident response tool stubs capture intent without execution

---

## Graph Architecture

```
START
  |
  v
classify_incident          (LLM: parse severity + category)
  |
  v
fan_out_router             (conditional: dispatches 3 Send() objects)
  |--- analyze_branch[security]        (parallel)
  |--- analyze_branch[performance]     (parallel)
  |--- analyze_branch[infrastructure]  (parallel)
  |         |
  v         v
aggregate_findings         (fan-in: merges 3 analysis results)
  |
  v
assess_quality             (LLM: scores findings 0.0-1.0)
  |
  v
quality_gate               (conditional)
  |--- [RETRY] score < 0.7 & attempts < 2: re-dispatch 3 parallel branches
  |--- [PASS]  otherwise: proceed
  |
  v
synthesize_report          (LLM: drafts incident report)
  |
  v
critique_report            (LLM: reviews report, verdict ACCEPT/REVISE)
  |
  v
critique_gate              (conditional)
  |--- [REVISE] verdict is REVISE & attempts < 2: loop to synthesize_report
  |--- [ACCEPT] otherwise: proceed
  |
  v
extract_actions            (LLM: parses ACTION/PARAMS blocks into tool intents)
  |
  v
risk_assessment            (LLM: evaluates final risk level)
  |
  v
final_summary              (LLM: generates comprehensive response)
  |
  v
END
```

---

## Graph Nodes

| Node | LLM Call | Purpose |
|------|----------|---------|
| `classify_incident` | Yes | Classifies incident severity (`critical`/`standard`/`low_priority`) and category (`security`/`performance`/`infrastructure`/`mixed`) |
| `analyze_branch` | Yes | Runs domain-specific analysis; spawned 3 times in parallel (one per category) |
| `aggregate_findings` | No | Pure logic node that merges the 3 parallel analysis results into a single summary |
| `assess_quality` | Yes | Scores the aggregated findings (0.0-1.0) and provides feedback |
| `synthesize_report` | Yes | Drafts a comprehensive incident report from analysis findings |
| `critique_report` | Yes | Reviews the draft report and issues an ACCEPT or REVISE verdict |
| `extract_actions` | Yes | Parses recommended actions and extracts structured tool-call intents |
| `risk_assessment` | Yes | Evaluates final risk level (`critical`/`high`/`medium`/`low`) |
| `final_summary` | Yes | Generates the comprehensive final response returned to the caller |

**Total LLM calls per query**: 9 (minimum, no retries) to 18 (maximum, both retry cycles exhausted).

---

## Conditional Edges

| Edge Function | Source Node | Behavior |
|--------------|-------------|----------|
| `fan_out_router` | `classify_incident` | Returns 3 `Send("analyze_branch", ...)` objects to spawn parallel analysis for security, performance, and infrastructure |
| `quality_gate` | `assess_quality` | If `quality_score < 0.7` and `quality_attempts < 2`, re-dispatches parallel branches; otherwise proceeds to `synthesize_report` |
| `critique_gate` | `critique_report` | If verdict contains "REVISE" and `synthesis_attempts < 2`, loops back to `synthesize_report`; otherwise proceeds to `extract_actions` |

---

## Incident Tools

Six decision-only tool stubs registered via `incident_tools.json`. These capture intent without execution:

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `create_ticket` | Creates a support/incident ticket | `title`, `priority` (P0-P3), `assignee_team`, `description` |
| `escalate_incident` | Escalates to the on-call team | `level` (L1-L3), `reason`, `urgency` |
| `notify_team` | Sends notification to a team | `team`, `channel` (slack/email/pager), `message` |
| `schedule_followup` | Schedules a follow-up action | `type` (review/postmortem/recheck), `delay_hours`, `description` |
| `check_system_status` | Checks system health | `system_name`, `check_type` (health/connectivity/performance/full_diagnostic) |
| `apply_mitigation` | Applies a mitigation action | `action`, `target_system`, `rollback_plan` |

Tool intents are captured via `ToolIntentBuffer` (shared with the `react_benchmark_agent` package) and extracted by the `extract_actions` node.

---

## State Schema

The `IncidentState` TypedDict tracks all data through the graph:

| Field | Type | Description |
|-------|------|-------------|
| `query` | `str` | Original incident description |
| `incident_id` | `str` | Unique incident identifier |
| `severity` | `str` | `critical`, `standard`, or `low_priority` |
| `category` | `str` | `security`, `performance`, `infrastructure`, or `mixed` |
| `classification_reasoning` | `str` | LLM reasoning for classification |
| `analysis_results` | `list[dict]` | Accumulated results from parallel branches (fan-in via `operator.add`) |
| `aggregated_findings` | `str` | Merged summary of all analyses |
| `quality_score` | `float` | Quality assessment score (0.0-1.0) |
| `quality_feedback` | `str` | Feedback explaining the quality score |
| `quality_attempts` | `int` | Quality gate retry counter (max 2) |
| `draft_report` | `str` | Synthesized incident report |
| `critique_feedback` | `str` | Critique verdict and feedback |
| `synthesis_attempts` | `int` | Synthesis retry counter (max 2) |
| `tool_calls_made` | `list[dict]` | Extracted tool-call intents |
| `recommended_actions` | `list[str]` | Recommended remediation actions |
| `risk_level` | `str` | `critical`, `high`, `medium`, or `low` |
| `risk_reasoning` | `str` | Risk assessment reasoning |
| `final_response` | `str` | Final comprehensive response |
| `messages` | `list` | NAT framework message list |

---

## Running Evaluations

### Step 1: Install NeMo Agent Toolkit from Source

```bash
cd /path/to/NeMo-Agent-Toolkit

uv sync --all-groups --extra most
```

### Step 2: Install the React Benchmark Agent (Dependency)

The complex LangGraph benchmark depends on `react_benchmark_agent` for shared tool intent stub utilities.

```bash
uv pip install -e examples/dynamo_integration/react_benchmark_agent/
```

### Step 3: Install the Complex LangGraph Benchmark

```bash
uv pip install -e examples/dynamo_integration/complex_langgraph_benchmark/
```

### Step 4: Configure the LLM Endpoint

Open the evaluation config file and update the `base_url` and `model_name` under the `llms` section to point to your model endpoint:

```yaml
# In configs/eval_config_full.yml
llms:
  dynamo_llm:
    _type: dynamo
    model_name: llama_70b          # Update to match your served model
    base_url: http://localhost:8099/v1  # Update to match your endpoint
    api_key: dummy
    temperature: 0.0
    max_tokens: 2048
```

### Step 5: Run the Full Evaluation (100 Scenarios)

```bash
nat eval --config_file examples/dynamo_integration/complex_langgraph_benchmark/configs/eval_config_full.yml
```

**Expected runtime**: ~30-60 minutes depending on Dynamo configuration and concurrency.

### Quick Validation (3 Scenarios)

For a faster smoke test, use the minimal config:

```bash
nat eval --config_file examples/dynamo_integration/complex_langgraph_benchmark/configs/eval_config_minimal.yml
```

**Expected runtime**: ~2-3 minutes.

### Evaluation Metrics

Both evaluation configs collect the following metrics:

| Evaluator | Description |
|-----------|-------------|
| `avg_llm_latency` | Average LLM call latency across all graph nodes |
| `avg_workflow_runtime` | Total workflow execution time per scenario |
| `avg_num_llm_calls` | Number of LLM calls per scenario (expected: 9-18) |
| `avg_tokens_per_llm_end` | Token efficiency per LLM response |

The full evaluation additionally enables advanced profiling: token uniqueness forecasting, workflow runtime forecasting, prompt caching prefix analysis, concurrency spike detection, and prediction trie generation.

---

## Configuration Reference

| Config File | Purpose | Dataset |
|-------------|---------|---------|
| `config_test_llm.yml` | Unit testing with `nat_test_llm` (no external services) | N/A |
| `config_dynamo_e2e_test.yml` | Basic Dynamo connectivity test | Single query |
| `config_dynamo_prefix.yml` | Dynamo with dynamic prefix headers for KV cache optimization | Single query |
| `eval_config_minimal.yml` | Smoke test evaluation (3 scenarios) | `complex_langgraph_benchmark_mini.json` |
| `eval_config_full.yml` | Full benchmark evaluation (100 scenarios) | `complex_langgraph_benchmark.json` |

### Single-Query Workflow Run

```bash
# Basic Dynamo connectivity test
nat run --config_file examples/dynamo_integration/complex_langgraph_benchmark/configs/config_dynamo_e2e_test.yml \
    --input "Database primary node is unresponsive, read replicas lagging"

# With prefix headers for KV cache optimization
nat run --config_file examples/dynamo_integration/complex_langgraph_benchmark/configs/config_dynamo_prefix.yml \
    --input "Critical: API gateway returning 502 errors, 40% of requests failing"
```

---

## Dataset

### Full Dataset (`complex_langgraph_benchmark.json`) - 100 Scenarios

Comprehensive IT incident scenarios spanning security breaches, performance degradation, infrastructure failures, and deployment issues.

### Mini Dataset (`complex_langgraph_benchmark_mini.json`) - 3 Scenarios

Minimal subset for quick validation and smoke testing.

### Scenario Format

```json
{
  "id": "incident_001",
  "question": "CRITICAL ALERT: auth-service cluster experiencing credential stuffing...",
  "ground_truth": "Create P0 ticket, escalate L3, notify team...",
  "metadata": {
    "expected_severity": "critical",
    "expected_category": "security",
    "expected_min_llm_calls": 9
  },
  "expected_tool_calls": [
    {"tool": "create_ticket", "parameters": {"priority": "P0", ...}},
    {"tool": "escalate_incident", "parameters": {"level": "L3", ...}}
  ]
}
```

---

## Troubleshooting

### Module Not Found: `complex_langgraph_benchmark`

Ensure both packages are installed:

```bash
uv pip install -e examples/dynamo_integration/react_benchmark_agent/
uv pip install -e examples/dynamo_integration/complex_langgraph_benchmark/
```

### Module Not Found: `react_benchmark_agent`

The complex benchmark depends on `react_benchmark_agent` for `ToolIntentBuffer`. Install it first:

```bash
uv pip install -e examples/dynamo_integration/react_benchmark_agent/
```

### Dynamo Connection Errors

Verify Dynamo is running:

```bash
curl http://localhost:8099/health
```

If not running, see the [Dynamo Setup Guide](../../../external/dynamo/README.md).

### File Not Found Errors

Run all `nat` commands from the **repository root**, not from the example directory:

```bash
cd /path/to/NeMo-Agent-Toolkit
nat eval --config_file examples/dynamo_integration/complex_langgraph_benchmark/configs/eval_config_full.yml
```

### Prediction Trie Path Errors

The `prediction_trie_path` in evaluation configs references a specific job ID. After your first run, update the path to match your actual output:

```yaml
prediction_trie_path: ./examples/dynamo_integration/complex_langgraph_benchmark/outputs/minimal_test/jobs/<your_job_id>/prediction_trie.json
```

Or remove the `prediction_trie_path` field to use only the static fallback prefix headers.
