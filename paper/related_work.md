# 2. Related Work

The evaluation of Large Language Models has evolved from static language modeling perplexity to comprehensive task-oriented benchmark suites and operational robustness metrics. This section reviews relevant literature in LLM evaluation, prompt engineering sensitivity, and local LLM serving infrastructure.

---

## 2.1 LLM Benchmarking & Evaluation Frameworks

Standard LLM benchmarks evaluate domain knowledge, reasoning, and code generation across standardized datasets. Frameworks such as MMLU [5], HELM [6], HumanEval [7], and GAIA [4] provide established task paradigms. However, traditional evaluation harnesses focus primarily on single-run top-1 accuracy under cloud API assumptions. They frequently lack built-in mechanisms for stochastic repeated-run consistency, text perturbation robustness, or systematic fault injection. Furthermore, standard evaluation tools assume high-availability cloud endpoints and are not designed to gracefully handle local hardware memory allocation failures or HTTP 500 server crashes.

---

## 2.2 LLM Robustness & Prompt Sensitivity

Prior studies demonstrate that LLMs are highly sensitive to prompt phrasing, system instructions, and minor typographical perturbations [8, 9]. Instruction-tuned models often default to verbose, conversational explanations containing Chain-of-Thought (CoT) derivations [10]. On exact-match benchmark datasets requiring concise string targets (such as numerical outputs or single entities), unconstrained conversational outputs lead to string matching failures [11]. Existing frameworks often address this by embedding ad-hoc string formatting hacks directly inside benchmark evaluation code. In contrast, our framework decouples prompting policies from evaluation adapters by injecting configurable system prompts at the experiment orchestration layer.

---

## 2.3 Local LLM Inference & Memory Management

The deployment of open-weights LLMs on local workstation hardware relies on optimized inference runtimes such as llama.cpp [12], vLLM [13], and Ollama [14]. Quantization techniques (e.g., 4-bit Q4_0 and 8-bit Q8_0) significantly reduce VRAM requirements, enabling 7B–9B models to execute on consumer hardware [15]. However, estimating total memory consumption during runtime—including KV cache allocation and context window overhead—remains challenging on Windows and Linux platforms [16]. When local servers encounter out-of-memory (OOM) conditions, they emit generic HTTP 500 internal server errors. Standard evaluation pipelines misclassify these as transient network failures and engage in repetitive, wasteful retry loops.

---

## 2.4 Summary of Architectural Differentiation

Table 1 highlights how the **LLM Reliability Ranking Framework** compares against existing evaluation tools:

| Feature / Capability | Standard Evaluation Harnesses | Local Inference Tools (Ollama/vLLM) | **Our Framework** |
|---|---|---|---|
| **Primary Metric** | Top-1 Accuracy | Tokens/Second | **Composite Reliability (Accuracy + Consistency + Robustness)** |
| **Memory Failure Handling** | Unhandled Crash / Hang | Server HTTP 500 Error | **Pre-flight Estimation + Non-Retryable Fast-Failing (0.0s Skip)** |
| **Prompt Policy** | Hardcoded / Fixed | Raw Input Forwarding | **Decoupled Configurable System Prompting** |
| **Fault & Perturbation** | None / Manual | None | **Built-in Perturbation & Fault Injection Managers** |
| **Artifact Traceability** | Summary JSON | Console Logs | **Canonical JSON Hierarchy + Checkpointing (`checkpoint.json`)** |
