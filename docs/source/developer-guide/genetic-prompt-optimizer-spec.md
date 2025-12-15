# Genetic Prompt Optimization Algorithm - Technical Specification

## Overview

The AIQ Toolkit includes a Genetic Algorithm (GA) based prompt optimization system that automatically evolves and improves prompts used in LLM-based workflows. This document provides a detailed technical specification of the algorithm implementation located at `src/nat/profiler/parameter_optimization/prompt_optimizer.py`.

## Design Goals

1. **Automated Prompt Improvement**: Evolve prompts to maximize task-specific metrics without manual tuning
2. **Multi-Objective Optimization**: Support optimization across multiple evaluation metrics simultaneously
3. **LLM-Guided Evolution**: Use LLMs to perform intelligent mutation and recombination of prompts
4. **Diversity Preservation**: Maintain population diversity to avoid premature convergence

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Genetic Prompt Optimizer                         │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐    │
│  │ Population   │──▶│  Evaluation  │──▶│ Fitness Scalarization│    │
│  │ Management   │   │    Engine    │   │   & Normalization    │    │
│  └──────────────┘   └──────────────┘   └──────────────────────┘    │
│         │                                         │                 │
│         │           ┌──────────────┐              │                 │
│         │◀──────────│   Selection  │◀─────────────┘                 │
│         │           │  (Tournament │                                │
│         │           │  / Roulette) │                                │
│         │           └──────────────┘                                │
│         │                  │                                        │
│         ▼                  ▼                                        │
│  ┌──────────────┐   ┌──────────────┐                               │
│  │   Elitism    │   │  Crossover   │                               │
│  │  (Top-k)     │   │ (LLM-based)  │                               │
│  └──────────────┘   └──────────────┘                               │
│         │                  │                                        │
│         │           ┌──────────────┐                               │
│         │           │   Mutation   │                               │
│         │           │ (LLM-based)  │                               │
│         │           └──────────────┘                               │
│         │                  │                                        │
│         └──────────────────┴────▶ Next Generation                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Data Structures

### Individual

Each individual in the population represents a candidate solution:

```python
@dataclass
class Individual:
    prompts: dict[str, str]           # param_name -> prompt text
    metrics: dict[str, float] | None  # evaluator_name -> average score
    scalar_fitness: float | None      # Combined fitness score
```

- **prompts**: Maps parameter names to their prompt text values
- **metrics**: Raw evaluation metric scores after workflow execution
- **scalar_fitness**: Normalized and combined fitness value used for selection

### SearchSpace (Prompt Configuration)

Prompts eligible for optimization are identified via the `SearchSpace` model:

```python
class SearchSpace:
    is_prompt: bool = False          # Flag indicating prompt optimization
    prompt: str | None = None        # Base prompt text to optimize
    prompt_purpose: str | None = None # Description of the prompt's objective
```

## Configuration Parameters

### PromptGAOptimizationConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | False | Enable GA-based prompt optimization |
| `prompt_population_init_function` | str | None | Function name for prompt mutation |
| `prompt_recombination_function` | str | None | Function name for crossover |
| `ga_population_size` | int | 24 | Number of individuals per generation |
| `ga_generations` | int | 15 | Total number of generations to evolve |
| `ga_offspring_size` | int | None | Offspring count (defaults to `pop_size - elitism`) |
| `ga_crossover_rate` | float | 0.8 | Probability of applying crossover [0.0-1.0] |
| `ga_mutation_rate` | float | 0.3 | Probability of mutation after crossover [0.0-1.0] |
| `ga_elitism` | int | 2 | Number of top individuals preserved unchanged |
| `ga_selection_method` | str | "tournament" | Selection strategy: "tournament" or "roulette" |
| `ga_tournament_size` | int | 3 | Contender count for tournament selection |
| `ga_parallel_evaluations` | int | 8 | Concurrent evaluation limit |
| `ga_diversity_lambda` | float | 0.0 | Diversity penalty strength (0 = disabled) |

### OptimizerConfig (General)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_metrics` | dict[str, OptimizerMetric] | Required | Metrics to optimize |
| `reps_per_param_set` | int | 3 | Evaluation repetitions per individual |
| `multi_objective_combination_mode` | str | "harmonic" | Score aggregation method |
| `target` | float | None | Early stopping target value |
| `output_path` | Path | None | Directory for results and checkpoints |

### OptimizerMetric

| Field | Type | Description |
|-------|------|-------------|
| `evaluator_name` | str | Name of the evaluation metric |
| `direction` | str | "maximize" or "minimize" |
| `weight` | float | Weight for weighted sum mode (default: 1.0) |

## Algorithm Flow

### Phase 1: Initialization

```
1. Discover prompt search space
   - Filter SearchSpace entries where is_prompt=True
   - Extract base prompts and their purposes

2. Validate configuration
   - Ensure eval_metrics is provided
   - Load mutation and recombination functions from workflow builder

3. Create initial population
   - Individual[0] = original prompts (baseline)
   - Individuals[1..N-1] = LLM-mutated variants (parallel creation)
```

### Phase 2: Main Evolution Loop

```
FOR each generation g in [1..generations]:

    2.1 EVALUATE population
        - Run workflow with each individual's prompts
        - Collect metric scores (averaged over reps_per_param_set runs)

    2.2 NORMALIZE scores
        - Per-metric min-max normalization across generation
        - Invert scores for "minimize" objectives

    2.3 SCALARIZE fitness
        - Combine normalized scores using configured mode
        - Apply diversity penalty if enabled

    2.4 SAVE checkpoint
        - Export best individual's prompts to JSON
        - Append generation history to CSV

    2.5 SELECT parents & CREATE offspring
        - Preserve elite individuals (elitism)
        - Select parents via tournament/roulette
        - Apply crossover and mutation operators

    2.6 FORM next generation
        - population = elites + offspring
```

### Phase 3: Finalization

```
1. Final evaluation of population
2. Select best individual by scalar_fitness
3. Export:
   - optimized_prompts.json (best prompts)
   - ga_history_prompts.csv (full optimization history)
```

## Fitness Evaluation

### Metric Normalization

Scores are normalized per-generation to ensure fair comparison:

```python
def _normalize_generation(individuals, metric_names, directions, eps=1e-12):
    for each metric m with direction d:
        vals = [ind.metrics[m] for ind in individuals]
        vmin, vmax = min(vals), max(vals)

        for each individual:
            if vmax - vmin < eps:
                score01 = 0.5  # All same, neutral score
            else:
                score01 = (val - vmin) / (vmax - vmin)

            if d == "minimize":
                score01 = 1.0 - score01  # Flip for minimization
```

### Scalarization Modes

The `multi_objective_combination_mode` determines how multiple metrics are combined:

| Mode | Formula | Best For |
|------|---------|----------|
| `harmonic` | `n / Σ(1/score_i)` | Balanced optimization, penalizes extremes |
| `sum` | `Σ(weight_i × score_i)` | Weighted prioritization of metrics |
| `chebyshev` | `min(score_i)` | Maximizing the worst-performing metric |

### Diversity Penalty

When `ga_diversity_lambda > 0`, duplicate prompts are penalized:

```python
penalty = diversity_lambda × (duplicate_count - 1)
adjusted_fitness = scalar_fitness - penalty
```

## Selection Methods

### Tournament Selection

```python
def _tournament_select(population, k):
    contenders = random.sample(population, k=min(k, len(population)))
    return max(contenders, key=lambda i: i.scalar_fitness)
```

- Selects `k` random individuals
- Returns the one with highest fitness
- Higher `k` increases selection pressure

### Roulette Wheel Selection

```python
def roulette_select(population):
    total = sum(max(ind.scalar_fitness, 0) for ind in population)
    r = random.random() * total
    cumulative = 0
    for ind in population:
        cumulative += max(ind.scalar_fitness, 0)
        if cumulative >= r:
            return ind
    return population[-1]
```

- Selection probability proportional to fitness
- Individuals with higher fitness are more likely to be selected

## Genetic Operators

### Mutation (LLM-Based)

The mutation operator uses an LLM to generate improved prompt variants:

```python
async def _mutate_prompt(original_prompt: str, purpose: str) -> str:
    return await init_fn.acall_invoke(
        PromptOptimizerInputSchema(
            original_prompt=original_prompt,
            objective=purpose,
            oracle_feedback=None,
        )
    )
```

**Mutation Prompt Template** (`src/nat/agent/prompt_optimizer/prompt.py`):

The LLM receives instructions to:
- Preserve the original objective and task
- Keep intent and critical instructions intact
- Apply improvement hints: clarity, structure, schema adherence, error handling
- Use mutation operators: tighten, reorder, constrain, harden, defuse, format-lock, example-ify

### Crossover (LLM-Based)

The crossover operator combines two parent prompts:

```python
async def _recombine_prompts(a: str, b: str, purpose: str) -> str:
    if recombine_fn is None:
        return random.choice([a, b])  # Fallback: uniform selection

    payload = {
        "original_prompt": a,
        "objective": purpose,
        "oracle_feedback": None,
        "parent_b": b
    }
    return await recombine_fn.acall_invoke(payload)
```

The recombination function instructs the LLM to:
- Combine strongest instructions from both parents
- Maintain variables and placeholders unchanged
- Produce a single, coherent child prompt

### Child Creation

```python
async def _make_child(parent_a, parent_b) -> Individual:
    child_prompts = {}
    for param, (base_prompt, purpose) in prompt_space.items():
        pa = parent_a.prompts[param]
        pb = parent_b.prompts[param]
        child = pa

        # Apply crossover with probability crossover_rate
        if random.random() < crossover_rate:
            child = await _recombine_prompts(pa, pb, purpose)

        # Apply mutation with probability mutation_rate
        if random.random() < mutation_rate:
            child = await _mutate_prompt(child, purpose)

        child_prompts[param] = child

    return Individual(prompts=child_prompts)
```

## Elitism

Top-performing individuals are preserved unchanged:

```python
if elitism > 0:
    elites = sorted(population, key=lambda i: i.scalar_fitness, reverse=True)[:elitism]
    next_population = [Individual(prompts=e.prompts.copy()) for e in elites]
```

This ensures the best solutions are never lost due to stochastic selection.

## Concurrency Model

The algorithm uses asyncio for parallel operations:

```python
# Concurrent population initialization
init_sem = asyncio.Semaphore(max_eval_concurrency)
tasks = [_create_random_individual() for _ in range(needed)]
individuals = await asyncio.gather(*tasks)

# Concurrent evaluation
sem = asyncio.Semaphore(max_eval_concurrency)
evaluated = await asyncio.gather(*[_evaluate(ind) for ind in unevaluated])
```

The semaphore limits concurrent operations to prevent resource exhaustion.

## Output Artifacts

### Per-Generation Checkpoints

`optimized_prompts_gen{N}.json`:
```json
{
  "param_name": ["optimized prompt text", "prompt purpose"]
}
```

### Final Outputs

`optimized_prompts.json`:
```json
{
  "param_name": ["best optimized prompt text", "prompt purpose"]
}
```

`ga_history_prompts.csv`:
| Column | Description |
|--------|-------------|
| `generation` | Generation number |
| `index` | Individual index within generation |
| `scalar_fitness` | Combined fitness score |
| `metric::MetricName` | Raw metric scores |

## Integration Example

### Configuration

```yaml
optimizer:
  output_path: "./optimization_results"
  eval_metrics:
    accuracy:
      evaluator_name: "Accuracy"
      direction: "maximize"
      weight: 1.0
    latency:
      evaluator_name: "Latency"
      direction: "minimize"
      weight: 0.5
  multi_objective_combination_mode: "harmonic"
  reps_per_param_set: 3

  prompt:
    enabled: true
    prompt_population_init_function: "prompt_init"
    prompt_recombination_function: "prompt_recombiner"
    ga_population_size: 24
    ga_generations: 15
    ga_crossover_rate: 0.8
    ga_mutation_rate: 0.3
    ga_elitism: 2
    ga_selection_method: "tournament"
    ga_tournament_size: 3
    ga_parallel_evaluations: 8
    ga_diversity_lambda: 0.1
```

### Defining Optimizable Prompts

```python
from nat.data_models.optimizable import OptimizableField, SearchSpace

class MyWorkflowConfig(FunctionBaseConfig):
    system_prompt: str = OptimizableField(
        default="You are a helpful assistant.",
        space=SearchSpace(
            is_prompt=True,
            prompt="You are a helpful assistant.",
            prompt_purpose="System prompt for guiding LLM behavior in QA tasks"
        )
    )
```

## Algorithmic Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Population Init | O(P × M) | O(P × S) |
| Evaluation | O(P × R × E) | O(P × M) |
| Selection | O(P × T) tournament, O(P) roulette | O(1) |
| Reproduction | O(P × M) | O(P × S) |

Where:
- P = population size
- M = number of prompt parameters
- R = reps_per_param_set
- E = evaluation cost per workflow run
- T = tournament size
- S = average prompt string length

## Best Practices

1. **Population Size**: Start with 20-30 individuals; increase for complex optimization landscapes
2. **Generations**: 10-20 generations typically sufficient; monitor convergence via history CSV
3. **Crossover Rate**: 0.7-0.9 recommended; promotes exploration
4. **Mutation Rate**: 0.2-0.4 recommended; balances exploration vs. exploitation
5. **Elitism**: 1-3 individuals; prevents loss of good solutions
6. **Diversity Penalty**: Enable (λ=0.05-0.2) if population converges prematurely
7. **Evaluation Reps**: 3-5 reps reduce noise from stochastic LLM outputs

## Limitations

1. **Computational Cost**: Each evaluation requires full workflow execution
2. **LLM Dependency**: Mutation/crossover quality depends on the optimizer LLM
3. **Local Optima**: GA may converge to local optima; consider diversity penalty
4. **Prompt Length**: Very long prompts may be truncated by LLMs during mutation

## References

- Implementation: `src/nat/profiler/parameter_optimization/prompt_optimizer.py`
- Configuration models: `src/nat/data_models/optimizer.py`
- Mutation prompts: `src/nat/agent/prompt_optimizer/prompt.py`
- LLM functions: `src/nat/agent/prompt_optimizer/register.py`
- Tests: `tests/nat/profiler/test_prompt_optimizer.py`
