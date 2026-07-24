# 1. Introduction

As Large Language Models (LLMs) transition from cloud-hosted API services to locally deployed edge and enterprise software systems, traditional evaluation methodologies focused exclusively on static top-1 accuracy are increasingly insufficient [1]. Real-world software systems require LLMs to exhibit operational **reliability**—a multidimensional attribute encompassing response consistency under repeated sampling, structural robustness against input text perturbations, fault tolerance during transient network or service disruptions, and strict compliance with local hardware resource constraints [2].

Evaluating open-weights LLMs in locally hosted runtime environments (such as Ollama, vLLM, or llama.cpp) introduces unique engineering and methodological challenges:

1. **Hardware Memory Allocation Bottlenecks**: Open-weights models varying from 7 billion to 70 billion parameters require substantial GPU Video RAM (VRAM) or unified system RAM. On heterogeneous host environments, executing models that exceed physical memory boundaries leads to non-recoverable allocation failures (e.g., HTTP 500 internal server errors) that can crash traditional benchmark runners or trigger wasteful retry loops [3].
2. **Prompt Sensitivity & Evaluation Alignment**: Instruction-tuned models are highly sensitive to prompt formatting. When presented with unconstrained questions, LLMs default to conversational Chain-of-Thought (CoT) derivations. On exact-match benchmarks such as General AI Assistants (GAIA) [4], unprompted reasoning preambles cause string comparison failure, reporting false-negative 0.0% accuracy despite correct internal reasoning.
3. **Reproducibility & Execution Tracing**: Benchmarking multi-model, multi-repetition experiment matrices requires deterministic seed management, immutable configuration hashing, structured error taxonomy, and resilient checkpointing to resume interrupted runs.

To address these challenges, we introduce the **LLM Reliability Ranking Framework**, a production-ready, extensible research framework designed to evaluate local LLM architectures, fast-fail unrecoverable resource bottlenecks, and produce canonical reliability rankings suitable for academic publication.

---

## 1.1 Research Objectives

This work addresses three primary research objectives:
- **O1 (Architecture)**: Design a modular, decoupled framework separating model interaction (`Agent`), benchmark evaluation (`Benchmark`), system prompting, and metric aggregation.
- **O2 (Robustness & Resource Efficiency)**: Develop a memory-aware execution engine capable of pre-flight matrix validation, detecting non-retryable memory allocation failures, and executing 0-second automatic model skipping without breaking batch execution pipelines.
- **O3 (Empirical Evaluation)**: Evaluate four local Ollama model architectures (`llama3.1:8b`, `mistral:7b`, `qwen2.5:7b`, `gemma2:9b`) under strict physical memory constraints on the GAIA benchmark.

---

## 1.2 Contributions

The principal contributions of this paper are as follows:

1. **Open-Source Research Framework**: A production-ready Python framework for benchmarking local LLM reliability across multiple standard adapters (GAIA, AgentBoard, SWEBench Lite).
2. **Memory-Aware Execution Engine**: A deterministic error-handling pipeline that classifies non-retryable memory allocation failures (`OllamaMemoryError`), eliminates redundant retries, logs formatted skip notifications, and preserves artifact generation.
3. **Decoupled Configurable System Prompting Architecture**: An experiment-level system prompt injection mechanism that aligns LLMs with strict short-answer evaluation criteria without polluting benchmark adapter logic.
4. **Empirical Case Study**: A comparative evaluation demonstrating that system prompt alignment elevates GAIA exact-match accuracy from 0.0% to 100.0% for executable models (`qwen2.5:7b`, `gemma2:9b`, `mistral:7b`), while cleanly isolating hardware-constrained models (`llama3.1:8b`).
