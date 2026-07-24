# 1. Introduction

As Large Language Models (LLMs) transition from centralized cloud-hosted API infrastructure to locally executed workstation and edge computing environments, traditional evaluation methodologies focused exclusively on static top-1 single-query accuracy become insufficient [1]. Real-world autonomous agents and embedded software systems require LLMs to exhibit operational **reliability**—a multidimensional attribute encompassing response consistency under repeated sampling, structural robustness against input text perturbations, fault tolerance during transient network or service disruptions, and compliance with physical hardware memory boundaries [2].

Executing open-weights LLMs in locally hosted runtime environments (such as Ollama, vLLM, or llama.cpp) introduces distinct engineering and methodological challenges:

1. **Hardware Memory Allocation Bottlenecks**: Open-weights models ranging from 1B to 70B parameters require substantial GPU Video RAM (VRAM) or unified system RAM. On resource-constrained host hardware, attempting to load models that exceed physical memory boundaries induces non-recoverable server allocation failures (e.g., HTTP 500 internal server errors). Standard benchmark runners often misclassify these as transient failures, engaging in redundant retry loops or hanging execution pipelines [3].
2. **Prompt Sensitivity & Metric Alignment**: Instruction-tuned language models exhibit high sensitivity to prompt formatting. When presented with unconstrained query prompts, LLMs default to verbose, conversational explanations containing Chain-of-Thought (CoT) preambles. On exact-match benchmarks such as General AI Assistants (GAIA) [4], unprompted conversational outputs cause string comparison failure, yielding false-negative 0.0% accuracy despite correct internal task solution logic.
3. **Execution Traceability & Reproducibility**: Evaluating multi-model, multi-repetition experiment matrices requires deterministic seed management, immutable configuration hashing, structured error taxonomies, and checkpointing mechanisms to resume interrupted benchmark executions.

To address these challenges, we present the **LLM Reliability Ranking Framework**, an open-source, extensible research framework designed to evaluate local LLM architectures, fast-fail unrecoverable resource bottlenecks, and produce verifiable reliability rankings.

## 1.1 Research Objectives

This work addresses three primary research objectives:
- **O1 (Architecture)**: Design a modular, decoupled software architecture separating model interaction (`Agent`), benchmark evaluation (`Benchmark`), system prompt injection, and metric aggregation.
- **O2 (Robustness & Resource Efficiency)**: Develop a memory-aware execution engine capable of pre-flight matrix validation, detecting non-retryable memory allocation failures (`OllamaMemoryError`), and executing 0.0-second automatic model skipping without crashing multi-model experiment runs.
- **O3 (Empirical Evaluation)**: Evaluate six local Ollama model architectures (`tinyllama:1.1b`, `phi3:3.8b`, `qwen2.5:7b`, `mistral:7b`, `llama3.1:8b`, `gemma2:9b`) under physical memory constraints across 270 execution trials on GAIA validation tasks.

## 1.2 Contributions

The principal contributions of this paper are as follows:

1. **Open-Source Research Harness**: A modular Python framework for benchmarking local LLM reliability across standard adapters (GAIA, AgentBoard, SWE-bench Lite).
2. **Memory-Aware Execution Engine**: A deterministic error-handling pipeline that classifies non-retryable memory allocation failures (`OllamaMemoryError`), eliminates redundant retries, logs formatted skip notifications, and preserves artifact generation.
3. **Decoupled Configurable System Prompting Architecture**: An experiment-level system prompt injection mechanism that aligns LLMs with strict short-answer evaluation criteria without polluting benchmark adapter logic.
4. **Empirical Evaluation & Case Study**: A comparative evaluation across 270 execution runs demonstrating model capability differentiation (`tinyllama:1.1b` at 80.0% vs 7B–9B models at 100.0%), low response latencies (2.72s–3.38s mean), and deterministic fast-skipping of memory-failed models.
