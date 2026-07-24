# 3. Methodology

This section details the design, software architecture, and operational mechanics of the framework.

## 3.1 Framework Architecture

The framework adopts an object-oriented architecture designed to isolate model interaction, benchmark evaluation, metric aggregation, and artifact persistence.

```
+-----------------------------------------------------------------------------------+
|                            Experiment Orchestrator                                |
|        (Spec Generator, Pre-Flight Validator, Multi-Run Matrix Scheduler)         |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                             Experiment Runner                                     |
|         (Run Queue Execution, Status Tracking, Result Aggregation)                |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                             Experiment Pipeline                                   |
|   +-----------------------+   +----------------------+   +--------------------+   |
|   |    Benchmark Adapter  |   |     Ollama Agent     |   | Perturbation/Fault |   |
|   | (GAIA, AgentBoard...) |   | (HTTP Client, Cache) |   |      Managers      |   |
|   +-----------+-----------+   +----------+-----------+   +---------+----------+   |
+---------------+--------------------------+-------------------------+--------------+
                |                          |                         |
                v                          v                         v
+-----------------------------------------------------------------------------------+
|                            Evaluation & Metric Engines                            |
|    (ExecutionRecord -> EvaluationRecord -> MetricRecord -> RankingRecord)         |
+-----------------------------------------------------------------------------------+
```
*Figure 1. Architectural component layout and dataflow pipeline.*

## 3.2 Ollama Integration & Memory Weight Unloading (`keep_alive: 0`)

The `_OllamaAdapter` maintains an OpenAI-compatible REST API client bound to `http://127.0.0.1:11434/v1`. To prevent GPU VRAM and system RAM accumulation across multi-model experiment execution batches, `OllamaAgent.shutdown()` issues an explicit HTTP POST call to `/api/generate` with the payload parameter `keep_alive: 0`. This forces the local Ollama daemon to unload active model weights from memory immediately upon completion of an experiment run.

## 3.3 Non-Retryable Exception Taxonomy & Fast-Failing

When an HTTP 500 error containing memory allocation failure keywords occurs during model initialization:
1. `_OllamaAdapter` catches the exception and raises `OllamaMemoryError` initialized with `is_transient = False`.
2. `BaseLLMAdapter.retry()` checks `is_non_retryable` and fails **immediately on attempt 1** without engaging retry loops.
3. The experiment runner logs a formatted skip notification, marks remaining tasks for that candidate model as failed in 0.0 seconds, and proceeds automatically to the next model in the matrix.

## 3.4 Decoupled System Prompt Architecture

System prompts are injected at the `ExperimentRunner` orchestration layer. For GAIA short-answer tasks, the injected system prompt is:

```text
You are solving a GAIA benchmark task.
Return ONLY the final answer.
Do not explain your reasoning.
Do not use markdown.
Output exactly the final answer.
```

This instruction prevents conversational preambles while keeping benchmark adapter code (`GAIAAdapter`) completely decoupled from prompting logic.

## 3.5 Mathematical Formulation of Reliability Metrics

The framework evaluates candidate models across four normalized metric dimensions:

1. **Success Rate ($S$)**: The proportion of task trials resulting in exact-match correctness:
   \[
   S = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{score}_i = 1.0)
   \]
   where $N$ is total task executions and $\mathbb{I}(\cdot)$ is the indicator function.

2. **Repeated-Run Consistency ($C$)**: The agreement rate of generated answers across identical seeds and repeated trials:
   \[
   C = \frac{1}{N} \sum_{i=1}^{N} \max_{y} \frac{\sum_{r=1}^{R} \mathbb{I}(y_{i,r} = y)}{R}
   \]
   where $R$ is the number of repetition trials for task $i$ and $y_{i,r}$ is the string output.

3. **Perturbation Robustness ($R$)**: The ratio of success rate under character/word perturbations relative to unperturbed baseline performance:
   \[
   R = \frac{S_{\text{perturbed}}}{S_{\text{baseline}}}
   \]

4. **Composite Reliability Score ($W$)**: A weighted linear aggregation of normalized dimension scores:
   \[
   W = w_1 S + w_2 C + w_3 R + w_4 F \quad \left(\sum_{j=1}^{4} w_j = 1.0\right)
   \]
   where $F$ represents fault tolerance under simulated network/latency perturbations.
