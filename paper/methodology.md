# 3. Methodology

This section details the design, architecture, and operational mechanics of the **LLM Reliability Ranking Framework**. The framework is engineered to benchmark large language models (LLMs) under deterministic, reproducible, and hardware-constrained execution environments.

---

## 3.1 Framework Architecture

The framework adopts a decoupled, object-oriented architecture designed to isolate model interaction, benchmark execution, metric aggregation, and artifact persistence.

```
+-----------------------------------------------------------------------------------+
|                            Experiment Orchestrator                                |
|        (Spec Generator, Pre-Flight Validator, Multi-Run Matrix Scheduler)         |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                             Experiment Runner                                     |
|         (Run Queue Execution, Status Tracking, Result Aggregation)               |
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
*Figure 1. High-level component architecture and dataflow pipeline of the LLM Reliability Ranking Framework.*

The core framework interfaces comprise:
- **`Agent` Interface**: Abstracts model interaction via standard `initialize()`, `run(task)`, `reset()`, and `shutdown()` methods.
- **`Benchmark` Interface**: Abstracts dataset loading (`load()`), task enumeration (`list_tasks()`), task execution (`run(agent, task)`), and answer scoring (`evaluate(execution)`).
- **`ExperimentPipeline`**: Coordinates the end-to-end execution lifecycle for a single (benchmark, agent, seed) experiment instance.
- **`ExperimentRunner`**: Orchestrates sequential or parallel execution queues, maintains checkpoint state, and manages failure recovery.
- **`ExperimentOrchestrator`**: Parses matrix specifications (JSON/YAML), conducts pre-flight validation, and generates executable experiment specifications (`ExperimentSpec`).

---

## 3.2 Component Interaction & Execution Lifecycle

The execution lifecycle follows a strict pipeline:

1. **Matrix Validation**: The `ExperimentOrchestrator` validates target benchmark registration, local dataset presence, provider registry resolution, and model reachability prior to run queue generation.
2. **Deterministic Scheduling**: The `Scheduler` generates a queue of immutable `RunDescriptor` instances, mapping each run to a derived seed calculated via SHA-256 hashing of the base seed and model identifier.
3. **Pipeline Initialization**: For each `RunDescriptor`, `ExperimentRunner` constructs a canonical `Configuration` object, instantiates the target agent via `AgentFactory`, and initializes the benchmark adapter.
4. **Execution & Artifact Collection**: `ExperimentPipeline` iterates through benchmark tasks, recording timing, prompt tokens, model output text, and environment metadata in canonical `ExecutionRecord` structures.
5. **Evaluation & Aggregation**: Execution records are evaluated to produce `EvaluationRecord` items, which are grouped by (benchmark, agent) pairs to compute `MetricRecord` statistics and generate `RankingRecord` tables.

---

## 3.3 Ollama Provider Integration

The framework integrates with local Ollama instances via a specialized provider adapter (`_OllamaAdapter`) and high-level agent wrapper (`OllamaAgent`).

### 3.1.1 HTTP Client Reuse & Parameter Injection
Rather than creating new HTTP connections per inference request, `_OllamaAdapter` maintains a persistent OpenAI-compatible client bound to the local server endpoint (`http://127.0.0.1:11434/v1`). Hyperparameters (`temperature`, `max_tokens`, `top_p`, `system_prompt`) are extracted from `config.metadata` and injected into `chat.completions.create()` requests.

### 3.1.2 Automatic Memory Unloading (`keep_alive = 0`)
To prevent GPU VRAM and system RAM accumulation across multi-model experiment batches, `OllamaAgent.shutdown()` issues an explicit POST call to `/api/generate` with parameter `keep_alive: 0`. This forces the local Ollama server to unload model weights from host RAM/VRAM immediately upon run completion.

---

## 3.4 Benchmark Execution Workflow

Benchmark adapters derive from `BaseBenchmarkAdapter` to provide standardized interfaces for heterogeneous datasets.

```text
[Load Dataset JSON / Hub] -> [Task Enumeration] -> [Agent Prompting] -> [Answer Extraction] -> [Normalization] -> [Exact Match Scoring]
```
*Figure 2. Benchmark execution and task evaluation workflow.*

- **GAIA Adapter (`GAIAAdapter`)**: Loads GAIA Level 1–3 multimodal questions from local JSON fixtures or HuggingFace Hub. Extracted model text is normalized via `normalize_gaia_answer()`, stripping leading/trailing whitespace and punctuation before performing string equivalence matching.
- **AgentBoard Adapter (`AgentBoardAdapter`)**: Evaluates multi-turn agent interaction trajectories.
- **SWEBench Lite Adapter (`SWEBenchLiteAdapter`)**: Evaluates software patch generation tasks.

---

## 3.5 Reliability Metric Formulation

The framework evaluates model performance across four core dimensions:

1. **Success Rate ($S$)**: The proportion of tasks completed correctly:
   $$S = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{score}_i = 1.0)$$

2. **Repeated-Run Consistency ($C$)**: The agreement rate of answers across $R$ independent stochastic repetitions with identical seeds:
   $$C = \frac{1}{N} \sum_{i=1}^{N} \max_{y} \frac{\sum_{r=1}^{R} \mathbb{I}(y_{i,r} = y)}{R}$$

3. **Perturbation Robustness ($R$)**: The performance retention under input text perturbations (e.g., whitespace variation, typos, rephrasing, prompt wrappers):
   $$R = \frac{S_{\text{perturbed}}}{S_{\text{baseline}}}$$

4. **Fault Tolerance ($F$)**: The resilience of execution under simulated system fault injection (e.g., network timeouts, rate limiting, context truncation).

5. **Composite Reliability Score ($W$)**: A weighted sum reflecting overall operational reliability:
   $$W = w_1 S + w_2 C + w_3 R + w_4 F \quad \text{where } \sum w_i = 1.0$$

---

## 3.6 Memory-Aware Model Execution & Fast-Failing

To operate robustly on resource-constrained hardware without hanging or crashing experiment runs, the framework incorporates pre-flight memory estimation and deterministic fast-failing error handling.

### 3.6.1 Pre-Flight Memory Estimation
Before executing an Ollama model, `estimate_model_memory()` queries `POST /api/show` to extract model parameters (`parameter_size`, `quantization_level`). Available system RAM is queried cross-platform via Windows `GlobalMemoryStatusEx` or Linux `/proc/meminfo`. If estimated model memory exceeds available memory, a warning is logged.

### 3.6.2 Non-Retryable Exception Taxonomy
The framework defines an explicit exception hierarchy (`exceptions.py`) with an `is_transient` property:

```text
ProviderError (Base)
 ├── Transient Errors (is_transient = True): ConnectionError, RateLimitError
 └── Non-Retryable Errors (is_transient = False):
      ├── OllamaMemoryError (System RAM / VRAM allocation failure)
      ├── OllamaModelNotFoundError (Model tag missing in local registry)
      ├── OllamaServerNotFoundError (Server connection lost)
      ├── AuthenticationError (Invalid API credentials)
      └── ResponseValidationError (Malformed JSON response)
```

When an HTTP 500 error containing memory allocation failure keywords (`"requires more system memory"`, `"out of memory"`, `"alloc"`) is encountered:
1. `_OllamaAdapter` raises `OllamaMemoryError` with `is_transient = False`.
2. `BaseLLMAdapter.retry()` checks `is_non_retryable` and fails **immediately on attempt 1** without engaging retry loops.

---

## 3.7 Automatic Model Skipping & Failure Classification

When an unrecoverable model-level error occurs during model initialization or task execution:

```text
[Task Exception Caught] -> [Classify Error Category] -> [Log Skip Notice] -> [Fast-Skip Remaining Tasks (0.0s)] -> [Generate Artifacts]
```
*Figure 3. Automatic model skipping and failure classification workflow.*

1. **Failure Classification**: Error text is categorized into standard failure reasons:
   - `"memory"`: System RAM / VRAM allocation failures.
   - `"model_unavailable"`: Unpulled model tags or missing provider binaries.
   - `"timeout"`: Execution time limit breaches.
   - `"network"`: Server disconnects or reset connections.
   - `"inference"`: Syntax or generation errors.

2. **Model Skip Protocol**:
   - The framework logs a warning:
     ```text
     Model <model_name> skipped.
     Reason: insufficient system memory.
     Continuing with next scheduled model.
     ```
   - All remaining tasks for that model are fast-skipped in 0.0 seconds with synthetic `ExecutionRecord` entries marked `[SKIPPED]` and `status="error"`.
   - The runner proceeds automatically to the next scheduled model in the matrix.

3. **Artifact Integrity**: `EvaluationRecord`, `MetricRecord`, and `RankingRecord` structures are computed so that failed models are accurately represented at 0% completion without halting batch execution.

---

## 3.8 Configurable Prompting Architecture

To support benchmarks requiring specific output formatting (e.g., short-answer exact string matching in GAIA) without embedding benchmark-specific prompt instructions inside benchmark adapters:

1. **Top-Level Configuration**: System prompts are specified in experiment JSON/YAML configurations under `system_prompt`:
   ```json
   {
     "system_prompt": "You are solving a GAIA benchmark task.\nReturn ONLY the final answer.\nDo not explain your reasoning.\nDo not use markdown.\nOutput exactly the final answer."
   }
   ```
2. **Clean Propagation**: `ExperimentOrchestrator` attaches `system_prompt` to `AgentSpec.metadata["system_prompt"]`, which `OllamaAgent` injects as a system role message into OpenAI `/v1/chat/completions` API payloads.
3. **Decoupled Design**: Benchmark adapters (`GAIAAdapter`) remain solely responsible for task loading and evaluation, while prompting strategies remain fully configurable at the experiment layer.

---

## 3.9 Evaluation & Artifact Serialization Workflow

All experiment artifacts are serialized into canonical JSON format using Pydantic v2 schemas:

```text
Configuration (Config Hash SHA-256)
  └── ExecutionRecord (Task Output, Latency, Failure Reason)
       └── EvaluationRecord (Score, Exact Match Result)
            └── MetricRecord (Success Rate, Consistency, Composite Score)
                 └── RankingRecord (Relative Model Rankings)
```
*Figure 4. Hierarchy of canonical serialized experiment artifacts.*

The `ResultManager` saves incremental checkpoints (`checkpoint.json`) after each completed run, ensuring experiment progress can be resumed seamlessly following interruption.
