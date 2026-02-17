# Linear LangGraph Benchmark — KV Cache Prefix Reuse

A strictly linear 7-node LangGraph benchmark for stress-testing NVIDIA Dynamo's KV cache prefix reuse optimizations. Unlike the complex benchmark (parallel branches, conditional edges), this benchmark uses a pure sequential topology where every successive LLM call extends the previous conversation as a strict prefix.

## Architecture: Enterprise Architecture Review Pipeline

**Domain**: Enterprise cloud architecture proposals reviewed against a comprehensive standards framework.

### Graph Topology (7 nodes, pure linear)

```
START -> architecture_intake -> component_deep_dive -> security_posture ->
         reliability_scaling -> cost_efficiency -> compliance_gaps ->
         executive_verdict -> END
```

No conditional edges, no branching, no retry cycles. Exactly **7 LLM calls per run**.

### KV Cache Reuse Strategy

Each node appends to a shared `messages` list, then calls `llm.ainvoke(messages)`:

| Node | Input Tokens (approx) | New Output |
|------|-----------------------|------------|
| 1. architecture_intake | ~8,500 (system + intake + proposal) | ~1,000 |
| 2. component_deep_dive | ~9,800 (all above + step prompt) | ~1,200 |
| 3. security_posture | ~11,300 (all above + step prompt) | ~1,000 |
| 4. reliability_scaling | ~12,600 (all above + step prompt) | ~1,000 |
| 5. cost_efficiency | ~13,900 (all above + step prompt) | ~800 |
| 6. compliance_gaps | ~15,000 (all above + step prompt) | ~1,000 |
| 7. executive_verdict | ~16,300 (all above + step prompt) | ~1,500 |

**Intra-run cache reuse**: Each call's input is a strict prefix-extension of the previous (100% prefix cache hit).

**Inter-run cache reuse**: The system prompt (~5,000 tokens) is identical across all evaluation runs.

## Quick Start

### Install

```bash
uv pip install -e examples/dynamo_integration/linear_langgraph_benchmark
```

### Run with Test LLM (no external services)

```bash
nat workflow run \
  --config_file examples/dynamo_integration/linear_langgraph_benchmark/src/linear_langgraph_benchmark/configs/config_test_llm.yml \
  --input "Review this microservices architecture for payment processing"
```

### Run with Dynamo

```bash
# Basic Dynamo (no prefix headers)
nat workflow run \
  --config_file examples/dynamo_integration/linear_langgraph_benchmark/src/linear_langgraph_benchmark/configs/config_dynamo.yml \
  --input "Review this microservices architecture for payment processing"

# Dynamo with prefix headers for KV cache optimization
nat workflow run \
  --config_file examples/dynamo_integration/linear_langgraph_benchmark/src/linear_langgraph_benchmark/configs/config_dynamo_prefix.yml \
  --input "Review this microservices architecture for payment processing"
```

### Run Evaluation

```bash
# Minimal (3 scenarios, ~2-3 min)
nat eval run \
  --config_file examples/dynamo_integration/linear_langgraph_benchmark/src/linear_langgraph_benchmark/configs/eval_config_minimal.yml

# Full (100 scenarios, ~30-60 min)
nat eval run \
  --config_file examples/dynamo_integration/linear_langgraph_benchmark/src/linear_langgraph_benchmark/configs/eval_config_full.yml
```

## Testing

```bash
pytest examples/dynamo_integration/linear_langgraph_benchmark/tests/
```

## Dataset

- **Full**: 100 architecture proposals across 5 architecture styles (microservices, monolithic, serverless, event-driven, hybrid) and 10 industries
- **Mini**: 3 representative proposals for smoke testing

To regenerate the dataset:

```bash
cd examples/dynamo_integration/linear_langgraph_benchmark
python generate_dataset.py
```

## Comparison with Complex Benchmark

| Feature | Complex Benchmark | Linear Benchmark |
|---------|------------------|-----------------|
| Nodes | 12 (9 graph nodes) | 7 |
| Topology | Parallel branches, conditional edges, cycles | Pure linear |
| LLM calls/run | 9-18 (variable) | 7 (fixed) |
| LLM input type | `llm.ainvoke(prompt_string)` | `llm.ainvoke(messages_list)` |
| KV cache pattern | Independent prompts | Prefix-extension |
| Intra-run reuse | Minimal | 100% prefix hit |
| Inter-run reuse | Limited | System prompt shared |
| prefix_total_requests | 12 | 7 |

## Configuration

### Dynamo Prefix Settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `prefix_total_requests` | 7 | Exactly 7 LLM calls per graph traversal |
| `prefix_osl` | HIGH | Detailed review output (~800-1,500 tokens per node) |
| `prefix_iat` | LOW | Sequential calls arrive quickly within a run |
