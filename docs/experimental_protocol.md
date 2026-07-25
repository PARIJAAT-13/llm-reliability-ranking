# Experimental Protocol: LLM Reliability Evaluation

## 1. Research Questions

**RQ1**: How do LLMs of different scales compare in information preservation under fault conditions (measured by ISR)?

**RQ2**: Which fault types (timeout, API failure, invalid response, tool failure, context truncation) cause the most information loss, and does this ranking hold across model scales?

**RQ3**: Does Information Survival Rate (ISR) provide information beyond existing metrics (success rate, consistency, robustness, fault tolerance)?

**RQ4**: Is there a significant degradation trend in ISR over sequential runs (temporal reliability)?

**RQ5**: How does task difficulty correlate with ISR across models?

## 2. Hypotheses

| Hypothesis | Direction | Test |
|-----------|-----------|------|
| H1: Larger models have higher ISR | μ_frontier > μ_medium > μ_small | Kruskal-Wallis + Nemenyi |
| H2: "context_truncation" has lowest ISR | ISR_truncation < all others | Friedman + Nemenyi |
| H3: ISR composite explains variance beyond success rate alone | ΔR² > 0 | Hierarchical regression |
| H4: Temporal ISR slope is negative for all models | μ_slope < 0 | One-sample t-test / sign test |
| H5: ISR decreases with task difficulty | ρ < 0 | Spearman correlation |

## 3. Model Selection

Three tiers selected for representativeness, popularity, and practical constraints:

| Tier | Models | Size Range | Source |
|------|--------|------------|--------|
| **Small** | GPT-3.5-Turbo, Llama-3.1-8B, Mistral-7B, Gemma-2-9B | 7B–20B | API / local |
| **Medium** | GPT-4o-mini, Llama-3.1-70B, Qwen-2.5-32B, Mixtral-8x7B | 32B–70B | API / local |
| **Large** | GPT-4o, Claude-3.5-Sonnet, Llama-3.1-405B, Gemini-1.5-Pro | 70B+ | API |

**Inclusion criteria:** Publicly available, chat-optimized, English-proficient, standard API or local inference path.

## 4. Fault Injection Matrix

Each model × task × fault combination is run `n = 30` times (baseline) + `n = 30` per fault type.

| Fault Type | Mechanism | Severity | Duration |
|------------|-----------|----------|----------|
| `timeout` | Simulated response delay exceeding threshold | LIGHT (2s), MODERATE (5s), SEVERE (10s) | Per-request |
| `api_failure` | Simulated HTTP 500 / connection error | LIGHT (1 retry), MODERATE (3 retries), SEVERE (permanent) | Per-request |
| `invalid_response` | Simulated malformed output (null, random tokens) | LIGHT (partial), MODERATE (mostly), SEVERE (complete) | Per-request |
| `tool_failure` | Simulated tool execution error | LIGHT (warning), MODERATE (partial result), SEVERE (crash) | Per-request |
| `context_truncation` | Simulated context window overflow | MODERATE (50% drop), SEVERE (75% drop) | Per-request |
| `gpu_oom` | Simulated GPU memory exhaustion | SEVERE | Per-request |
| `runtime_crash` | Simulated process termination | CRITICAL | Per-request |

## 5. Sample Size Determination

Based on pilot data (d = 0.5 medium effect, α = 0.05, power = 0.80):

- **Within-model** (fault × baseline): `n = 30` per condition (Mann-Whitney)
- **Between-model** (model A × model B): `n = 50` per model (Kruskal-Wallis)
- **Temporal ISR**: `n = 10` windows × 6 observations each

**Total evaluations:** 12 models × 30 tasks × (1 baseline + 7 fault types) × 30 reps ≈ 864,000

## 6. Statistical Analysis Plan

| Analysis | Primary Test | Effect Size | Correction |
|----------|-------------|-------------|------------|
| Model × ISR (overall) | Kruskal-Wallis | Eta-squared (ε²) | — |
| Post-hoc model pairs | Nemenyi test | — | — |
| Fault type × ISR | Friedman test | Kendall's W | — |
| Post-hoc fault pairs | Nemenyi test | — | — |
| Task difficulty × ISR | Spearman correlation | ρ | — |
| ISR vs existing metrics | Spearman correlation | ρ | Bonferroni |
| Temporal degradation | One-sample t-test | Hedges' g | — |
| Model × temporal slope | Kruskal-Wallis | ε² | — |

## 7. Sensitivity Analyses

1. **ISR bin count robustness**: Repeat ISR calculation with n_bins = {5, 10, 20}
2. **ISR alpha sensitivity**: Repeat composite with α = {0.4, 0.6, 0.8}
3. **Bootstrap CI coverage**: Verify nominal 95% CI coverage via simulation
4. **Outlier exclusion**: Compare results with and without ±3 SD score outliers

## 8. Reproducibility Measures

- All random seeds recorded per evaluation
- Configuration hash for every run
- Software environment captured via reproducibility checklist
- Docker image / requirements.txt pinned
- Raw evaluation records archived as JSONL
- Analysis scripts versioned alongside data
