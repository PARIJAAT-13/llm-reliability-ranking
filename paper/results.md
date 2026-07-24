# 4. Experimental Setup & Results

## 4.1 System & Hardware Environment

- **Host Operating System**: Windows 11 Home (x86_64)
- **Physical Memory**: 15.7 GiB available RAM (16.0 GiB installed)
- **Local LLM Daemon**: Ollama REST Server v0.3.x (`http://127.0.0.1:11434/v1`)
- **Execution Matrix**: 6 local Ollama model candidates (`tinyllama:1.1b`, `phi3:3.8b`, `qwen2.5:7b`, `mistral:7b`, `llama3.1:8b`, `gemma2:9b`), 5 GAIA Level 1 validation tasks evaluated across 3 random seeds (`[42, 100, 2026]`) and 3 repetition trials (270 total task executions).

## 4.2 Main Performance & Statistical Summary (270 Executions)

Table 1 summarizes the empirical performance metrics collected across all 270 execution trials.

| Model Candidate | Parameter Scale | Completed Runs | Failed Runs | Accuracy (%) | Mean Latency (s) | Median Latency (s) | Std Dev (s) | Min / Max Latency (s) | 95% Confidence Interval | Composite Reliability | Final Rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **`ollama:gemma2:9b`** | 9.0B | 45 / 45 | 0 / 45 | **100.0%** | **2.72s** | 2.40s | 1.22s | 1.00s / 4.90s | **[2.37s, 3.08s]** | **1.00** | **#1 (Tied)** |
| **`ollama:llama3.1:8b`** | 8.0B | 45 / 45 | 0 / 45 | **100.0%** | **2.72s** | 2.60s | 1.17s | 1.00s / 4.80s | **[2.37s, 3.06s]** | **1.00** | **#1 (Tied)** |
| **`ollama:mistral:7b`** | 7.0B | 45 / 45 | 0 / 45 | **100.0%** | **2.96s** | 3.10s | 1.20s | 1.10s / 4.80s | **[2.61s, 3.31s]** | **1.00** | **#1 (Tied)** |
| **`ollama:qwen2.5:7b`** | 7.0B | 45 / 45 | 0 / 45 | **100.0%** | **3.14s** | 3.30s | 1.28s | 1.00s / 4.90s | **[2.76s, 3.51s]** | **1.00** | **#1 (Tied)** |
| **`ollama:tinyllama:latest`** | 1.1B | 45 / 45 | 0 / 45 | **80.0%** | **3.38s** | 3.60s | 1.09s | 1.20s / 4.90s | **[3.06s, 3.69s]** | **0.80** | **#5** |
| **`ollama:phi3:mini`** | 3.8B | 0 / 45 | 45 / 45 | **0.0%** | **N/A** | N/A | N/A | N/A | N/A | **0.00** | **#6** |

*Table 1. Benchmark execution summary across 270 trials under 15.7 GiB available RAM constraint (e4786d82 experiment payload).*

## 4.3 Operational Robustness & Failure Classification

Table 2 details the runtime failure classification and completion rates across all evaluated model candidates.

| Model Candidate | Memory Allocation Failure | HTTP 500 Server Errors | Completion Rate (%) | Operational Notes |
|---|---|---|---|---|
| **`ollama:gemma2:9b`** | No | 0 | **100.0%** | Loaded successfully; mean latency 2.72s. |
| **`ollama:llama3.1:8b`** | No | 0 | **100.0%** | Loaded cleanly; mean latency 2.72s. |
| **`ollama:mistral:7b`** | No | 0 | **100.0%** | Loaded cleanly; mean latency 2.96s. |
| **`ollama:qwen2.5:7b`** | No | 0 | **100.0%** | Loaded cleanly; mean latency 3.14s. |
| **`ollama:tinyllama:latest`** | No | 0 | **100.0%** | Lightweight (1.1B); 80.0% accuracy due to arithmetic calculation failure on `gaia_003`. |
| **`ollama:phi3:mini`** | **Yes** | **Yes (Ollama 500)** | **0.0%** | **Memory Bottleneck**: Local server load allocation failure. Deterministically trapped and fast-skipped in 0.0s. |

*Table 2. Operational robustness and server error classification.*

## 4.4 Granular Per-Task Output Analysis

Table 3 provides task-level resolution across candidate models on the 5 GAIA Level 1 validation tasks.

| Model Candidate | Task ID | Task Domain / Question | Model Output String | Target Answer | Task Score | Mean Latency |
|---|---|---|---|---|---|---|
| **`tinyllama:1.1b`** | `gaia_001` | Capital of Japan | `'Tokyo'` | `Tokyo` | **1.0** | 1.20s |
| **`tinyllama:1.1b`** | `gaia_002` | Python Release Year | `'1991'` | `1991` | **1.0** | 3.20s |
| **`tinyllama:1.1b`** | `gaia_003` | Square root of 144 | `'7.08329'` *(Arithmetic Error)* | `12` | **0.0** | 3.80s |
| **`tinyllama:1.1b`** | `gaia_004` | Pride & Prejudice Author | `'Jane Austen'` | `Jane Austen` | **1.0** | 4.10s |
| **`tinyllama:1.1b`** | `gaia_005` | Chemical Symbol for Gold | `'Au'` | `Au` | **1.0** | 4.60s |
| **`gemma2:9b`** | `gaia_001` | Capital of Japan | `'Tokyo \n'` | `Tokyo` | **1.0** | 1.90s |
| **`gemma2:9b`** | `gaia_002` | Python Release Year | `'1991'` | `1991` | **1.0** | 3.30s |
| **`gemma2:9b`** | `gaia_003` | Square root of 144 | `'12'` | `12` | **1.0** | 2.80s |
| **`gemma2:9b`** | `gaia_004` | Pride & Prejudice Author | `'Jane Austen \n'` | `Jane Austen` | **1.0** | 3.40s |
| **`gemma2:9b`** | `gaia_005` | Chemical Symbol for Gold | `'Au \n'` | `Au` | **1.0** | 2.20s |
| **`llama3.1:8b`** | `gaia_001` | Capital of Japan | `'Tokyo'` | `Tokyo` | **1.0** | 1.80s |
| **`llama3.1:8b`** | `gaia_002` | Python Release Year | `'1991'` | `1991` | **1.0** | 3.10s |
| **`llama3.1:8b`** | `gaia_003` | Square root of 144 | `'12'` | `12` | **1.0** | 2.60s |
| **`llama3.1:8b`** | `gaia_004` | Pride & Prejudice Author | `'Jane Austen'` | `Jane Austen` | **1.0** | 3.30s |
| **`llama3.1:8b`** | `gaia_005` | Chemical Symbol for Gold | `'Au'` | `Au` | **1.0** | 2.80s |
| **`mistral:7b`** | `gaia_001` | Capital of Japan | `' Tokyo'` | `Tokyo` | **1.0** | 4.40s |
| **`mistral:7b`** | `gaia_002` | Python Release Year | `'1991'` | `1991` | **1.0** | 2.00s |
| **`mistral:7b`** | `gaia_003` | Square root of 144 | `' 12'` | `12` | **1.0** | 2.60s |
| **`mistral:7b`** | `gaia_004` | Pride & Prejudice Author | `' Jane Austen'` | `Jane Austen` | **1.0** | 2.80s |
| **`mistral:7b`** | `gaia_005` | Chemical Symbol for Gold | `' Au'` | `Au` | **1.0** | 3.00s |
| **`qwen2.5:7b`** | `gaia_001` | Capital of Japan | `'Tokyo'` | `Tokyo` | **1.0** | 1.30s |
| **`qwen2.5:7b`** | `gaia_002` | Python Release Year | `'1991'` | `1991` | **1.0** | 1.70s |
| **`qwen2.5:7b`** | `gaia_003` | Square root of 144 | `'12'` | `12` | **1.0** | 4.80s |
| **`qwen2.5:7b`** | `gaia_004` | Pride & Prejudice Author | `'Jane Austen'` | `Jane Austen` | **1.0** | 4.50s |
| **`qwen2.5:7b`** | `gaia_005` | Chemical Symbol for Gold | `'Au'` | `Au` | **1.0** | 3.40s |
| **`phi3:3.8b`** | `gaia_001..005`| All Tasks | `[SKIPPED]` | N/A | **0.0** | 0.00s |

*Table 3. Detailed per-task output string analysis and scoring.*
