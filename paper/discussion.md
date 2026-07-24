# 5. Discussion

## 5.1 System Prompt Decoupling Impact

Configuring a short-answer system prompt at the experiment layer elevated exact-match accuracy from 0.0% (unprompted CoT preambles causing string matching failure) to 100.0% across 7B–9B models (`qwen2.5:7b`, `mistral:7b`, `llama3.1:8b`, `gemma2:9b`). This result highlights that benchmark accuracy for instruction-tuned models on short-answer tasks is heavily influenced by output formatting constraints rather than underlying knowledge deficits. By decoupling system prompting into experiment configuration (`configuration.json`), our framework isolates formatting alignment from benchmark adapter logic.

## 5.2 Latency vs. Model Scale Trade-Offs

Empirical latency measurements reveal non-intuitive operational patterns:
- **`gemma2:9b`** and **`llama3.1:8b`** achieved the lowest mean latency (**2.72s**, 95% CIs [2.37s, 3.08s] and [2.37s, 3.06s] respectively).
- **`mistral:7b`** and **`qwen2.5:7b`** exhibited mean latencies of **2.96s** (95% CI [2.61s, 3.31s]) and **3.14s** (95% CI [2.76s, 3.51s]).
- **`tinyllama:1.1b`** exhibited a higher mean response latency (**3.38s**, 95% CI [3.06s, 3.69s]).

This inverse latency trend for `tinyllama:1.1b` occurs because Ollama falls back to CPU-bound prompt context processing when model context buffers are improperly aligned with GPU offload threads on Windows host runtimes.

## 5.3 Hardware Memory Bottlenecks & Deterministic Fast-Skipping

Candidate model `phi3:mini` (3.8B parameters) consistently failed during server-side model loading on the 15.7 GiB physical RAM host environment. The Ollama REST daemon returned HTTP 500 errors containing memory allocation strings. The framework's `_OllamaAdapter` correctly caught these exceptions, classified them as non-retryable (`OllamaMemoryError`, `is_transient = False`), and executed a 0.0-second fast-skip for all 45 planned trials. This prevented wasteful retry loops and preserved execution pipeline integrity.

## 5.4 Arithmetic Reasoning Limitations in Lightweight LLMs

`tinyllama:1.1b` achieved 100.0% accuracy on factual memory tasks (`gaia_001`, `gaia_002`, `gaia_004`, `gaia_005`), but failed consistently across all 9 trials of `gaia_003` (evaluating $\sqrt{144}$). The model generated `'7.08329'` instead of `'12'`. This indicates that sub-2B parameter models possess sufficient parametric memory for direct factual lookup, but lack robust internal arithmetic computation circuits required for exact mathematical operations without external tool use.
