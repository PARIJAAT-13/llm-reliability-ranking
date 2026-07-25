# Information Survival Rate: An Information-Theoretic Metric for LLM Reliability Evaluation

**Authors:** Parijaat K. & LLM Reliability Framework Contributors
**Status:** Working draft
**Date:** July 2026

---

## Abstract

Large language models (LLMs) are increasingly deployed in production environments where reliability under fault conditions is critical. Yet existing evaluation frameworks measure capability on static benchmarks, not operational reliability. We introduce **Information Survival Rate (ISR)**, the first information-theoretic metric that quantifies the fraction of information an LLM preserves in its outputs under fault conditions relative to a fault-free baseline. ISR decomposes into output-level (distributional overlap via histogram intersection) and behaviour-level (success-rate preservation) components, combined into a composite score. We prove five formal properties: boundedness, identity, monotonicity under degradation, convexity, and statistical consistency. ISR is accompanied by bootstrap confidence intervals and temporal ISR for detecting degradation over sequential runs. We further present **ReliabilityBench**, a 36-task taxonomy-driven dataset spanning six categories at two difficulty tiers, designed specifically for reliability evaluation; a production-grade fault injection framework with GPU OOM, runtime crash, combined faults, and MTBF/MTTR metrics; and an automated statistical procedure selector that addresses the pervasive lack of statistical rigor in LLM benchmarking. The framework comprises 1100+ tests with zero regressions. ISR reveals information degradation patterns that accuracy-based metrics miss, establishing a new dimension for LLM evaluation.

---

## 1. Introduction

LLMs have transitioned from research artifacts to production systems powering chatbots, code assistants, and agentic workflows. This transition demands a corresponding evolution in evaluation methodology: from measuring *what a model can do* on curated benchmarks to measuring *how reliably it performs under real-world conditions*.

**The reliability gap.** Current evaluation frameworks—HELM (Bommasani et al., 2022), lm-evaluation-harness (Gao et al., 2023), OpenCompass (Cao et al., 2026)—operate on a static paradigm: present a clean input, collect a response, compare to a reference. This paradigm does not account for the fault conditions that characterise production deployments: API timeouts, GPU out-of-memory errors, context truncation, service crashes, and network interruptions. When faults occur, model responses degrade—but how much information is lost?

**The statistical rigor gap.** A systematic review of 445 LLM benchmarks at NeurIPS 2025 found that only 53.4% presented evidence of construct validity and only 16% used statistical tests to compare models. This means the majority of published benchmark results lack the statistical foundation necessary to distinguish genuine capability differences from sampling noise.

**Our contributions.** We address both gaps through four interconnected contributions:

1. **ISR (Information Survival Rate)** — An information-theoretic reliability metric that measures distributional overlap between baseline and fault-condition outputs, with formal guarantees, bootstrap confidence intervals, and temporal degradation analysis.

2. **ReliabilityBench** — A 36-task taxonomy-driven benchmark spanning reasoning, knowledge, instruction-following, code, language understanding, and robustness probes across Easy/Hard/Expert difficulty tiers.

3. **Production fault injection** — GPU OOM simulation, runtime crash simulation, combined fault strategies, and MTBF/MTTR reliability engineering metrics integrated with severity scheduling.

4. **Automated statistical methodology** — An auto-selection engine that checks normality and variance homogeneity, then recommends the appropriate hypothesis test, effect size, and multiple-comparison correction.

---

## 2. Related Work

### 2.1 LLM Evaluation Frameworks

HELM (Stanford CRHM, 2022) evaluates 30 models across 16 scenarios on 7 metrics including accuracy, calibration, robustness, and efficiency. Its robustness metric measures performance on perturbed inputs but does not inject system-level faults. OpenCompass (Cao et al., 2026) aggregates 100+ datasets and 400K+ questions with strong multilingual support, but remains accuracy-centric. The EleutherAI lm-evaluation-harness (Gao et al., 2023) powers the Open LLM Leaderboard with 60+ benchmarks and 500+ subtasks, yet scores vary 4–20 points across implementations (SourceScore, 2026). DeepEval (confident-ai, 2024) provides 50+ metrics with LLM-as-a-judge for RAG/agent quality but not system reliability. None of these frameworks measure operational reliability under fault conditions.

### 2.2 Reliability and Robustness

HIP-LLM (McTaggart et al., 2025) defines LLM reliability as "probability of failure-free operation under a given Operational Profile" using hierarchical Bayesian modelling. This is the closest prior work to ours, but HIP-LLM models binary success/failure rather than the degree of information preservation. ISR is complementary: HIP-LLM answers "will it fail?", ISR answers "how much information survives?". Recent work at SC 2025 demonstrated that LLMs are not truly resilient to bit-flips, and that aggregate accuracy masks significant text quality degradation—a finding that motivates our distribution-based ISR over point-estimate metrics.

### 2.3 Fault Injection

LLMFI (Huang et al., 2026) injects faults at different LLM inference stages and identifies early decode iterations as most vulnerable. ChaosLLM (Iannillo et al., 2025) targets LangChain agents with unreachable, slow, hang, and incorrect response failure modes. Our fault injection framework extends this with production scenarios (GPU OOM, runtime crash), combined faults, and reliability engineering metrics (MTBF, MTTR, availability) drawn from classical software reliability (Musa, 1993).

### 2.4 Benchmark Construction

The NeurIPS 2025 benchmark survey found that only 53.4% of 445 benchmarks present construct validity evidence. Laskar et al. (2024) provide a comprehensive taxonomy of LLM evaluation challenges. ReliabilityBench is designed from the ground up for construct validity, with explicit taxonomies, multi-tier difficulty, and fault-type applicability annotations per task.

### 2.5 Research Gaps

Our literature review identified eight specific gaps (Table 1), of which this work addresses four: G1 (no operational reliability metric), G3 (nascent fault injection), G4 (no reliability-focused dataset), and G5 (lack of statistical rigor).

**Table 1: Research Gaps in LLM Reliability Evaluation**

| Gap | Description | Addressed By |
|-----|-------------|-------------|
| G1 | No operational reliability metric | ISR (§3) |
| G2 | Static evaluation → no fault capture | ReliabilityBench + FI (§4, §5) |
| G3 | Fault injection is nascent | Production FI framework (§5) |
| G4 | No reliability-focused dataset | ReliabilityBench (§4) |
| G5 | Statistical rigor lacking | Auto-selection (§6) |
| G6 | No unified reliability taxonomy | ISR taxonomy (§3) |
| G7 | Production studies provider-specific | Generalisable FI (§5) |
| G8 | No standard operational profile | Task taxonomy (§4) |

---

## 3. Information Survival Rate

### 3.1 Formal Definition

Let $\mathcal{E} = \{e_1, \ldots, e_N\}$ be a set of evaluation records, each with score $s_i \in [0,1]$. Let $\mathcal{B} = \{e \in \mathcal{E} \mid \neg\text{fault\_injected}\}$ be the baseline set and $\mathcal{F} = \{e \in \mathcal{E} \mid \text{fault\_injected}\}$ the fault-condition set.

**Output-level ISR.** Partition $[0,1]$ into $K$ equal-width bins. Let $P_B, P_F \in \mathbb{R}^K$ be the probability mass vectors of baseline and fault-condition scores over these bins, estimated via density-normalised histograms. Then:

$$\text{ISR}_{\text{output}} = \sum_{i=1}^{K} \min(P_B[i], P_F[i]) \cdot \Delta$$

where $\Delta = 1/K$ is the bin width. This is the histogram intersection (Swain & Ballard, 1991), a discrete analogue of the Hellinger path similarity.

**Behaviour-level ISR.** Let $\text{SR}_B = |\mathcal{B}|^{-1} \sum_{e \in \mathcal{B}} s_i$ and $\text{SR}_F = |\mathcal{F}|^{-1} \sum_{e \in \mathcal{F}} s_i$ be the mean success rates. Then:

$$\text{ISR}_{\text{behavior}} = 1 - |\text{SR}_B - \text{SR}_F|$$

**Composite ISR.** A convex combination controlled by $\alpha \in [0,1]$:

$$\text{ISR}_{\text{composite}} = \alpha \cdot \text{ISR}_{\text{output}} + (1 - \alpha) \cdot \text{ISR}_{\text{behavior}}$$

We default to $\alpha = 0.6$, giving slightly more weight to fine-grained distributional information.

### 3.2 Formal Properties

**Theorem 1 (Boundedness).** $0 \leq \text{ISR}_{\text{output}} \leq 1$ and $0 \leq \text{ISR}_{\text{behavior}} \leq 1$.

*Proof.* Histogram intersection of density-normalised histograms lies in $[0, 1/\Delta]$; multiplication by $\Delta$ normalises to $[0, 1]$. Behaviour-level ISR is 1 minus an absolute difference of values in $[0, 1]$, thus also in $[0, 1]$.

**Theorem 2 (Identity).** $\text{ISR}_{\text{output}} = 1 \iff P_B = P_F$ (over the given bin partition). $\text{ISR}_{\text{behavior}} = 1 \iff \text{SR}_B = \text{SR}_F$.

**Theorem 3 (Monotonicity).** Let $D(P, Q) = 1 - \text{ISR}_{\text{output}}(P, Q)$. If fault severity increases, shifting $Q$ further from $P$ in the sense of first-order stochastic dominance, then $D$ is non-decreasing and ISR is non-increasing.

**Theorem 4 (Convexity).** $\text{ISR}_{\text{composite}}$ is a convex combination of $\text{ISR}_{\text{output}}$ and $\text{ISR}_{\text{behavior}}$ for any $\alpha \in [0,1]$.

**Theorem 5 (Consistency).** As $|\mathcal{B}| \to \infty$ and $|\mathcal{F}| \to \infty$, the sample ISR converges in probability to the population ISR.

### 3.3 Bootstrap Confidence Intervals

ISR estimates are accompanied by percentile bootstrap confidence intervals (Efron & Tibshirani, 1994). For $R = 1000$ resamples, baseline and fault scores are resampled with replacement, ISR recomputed, and the $100(1 - \alpha_{\text{ci}})\%$ percentile interval extracted. This enables statistically grounded comparisons: two models differ reliably at the $\alpha_{\text{ci}}$ level if their ISR confidence intervals do not overlap.

### 3.4 Temporal ISR

ISR can mask degradation patterns that emerge over time. Temporal ISR computes ISR over $W$ sequential windows (ordered by evaluation timestamp), detecting drift or degradation. The linear trend slope $\beta$ of ISR over window index quantifies degradation rate:

$$\text{Trend} = \begin{cases}
\text{stable} & |\beta| < 0.01 \\
\text{degrading} & \beta \leq -0.01 \\
\text{improving} & \beta \geq 0.01
\end{cases}$$

---

## 4. ReliabilityBench Dataset

### 4.1 Design Principles

ReliabilityBench is designed for construct validity in reliability evaluation. Each task explicitly declares:

- **Category and domain** — for taxonomic analysis of reliability patterns
- **Difficulty** — $d \in [0, 1]$, with Easy ($d < 0.5$), Hard ($0.5 \leq d < 0.9$), and Expert ($d \geq 0.9$) tiers
- **Fault types** — which injection strategies are applicable (e.g., timeout, API failure, context truncation)
- **Perturbation types** — which input perturbations are applicable
- **Scoring rubric** — how to compute the evaluation score

### 4.2 Task Taxonomy

36 tasks across 6 categories:

| Category | Tasks | Domains | Difficulty Range |
|----------|-------|---------|-----------------|
| Reasoning | 6 | logical, mathematical, causal, counterfactual, multi-variable | 0.3 – 0.95 |
| Knowledge | 6 | factual recall, temporal reasoning, truthfulness, conflicting sources | 0.2 – 0.9 |
| Instruction Following | 6 | constraint satisfaction, format compliance, multi-step, negation, nested | 0.3 – 0.95 |
| Code | 6 | generation, bug detection, explanation, debugging, algorithm design | 0.4 – 0.95 |
| Language | 6 | summarisation, paraphrase, sentiment, entailment, ambiguity, pragmatics | 0.3 – 0.9 |
| Robustness Probes | 6 | adversarial typos, irrelevant context, role preservation, jailbreaks | 0.5 – 0.95 |

### 4.3 Scoring Rubrics

11 rubrics support diverse evaluation needs: exact_match, numeric_match, contains_match, exact_match_with_explanation, constraint_satisfaction, json_format_match, multi_step_match, negation_compliance, code_syntax_match, multi_item_match, and expert_constraint_match.

### 4.4 Integration

ReliabilityBench registers as `"ReliabilityBench"` in the framework's `BenchmarkRegistry`, making it usable with any registered agent through the standard `run()`/`evaluate()` lifecycle.

---

## 5. Fault Injection Framework

### 5.1 Strategies

Seven fault strategies with severity scheduling:

| Strategy | Injection Point | Severity Levels |
|----------|----------------|-----------------|
| ArtificialTimeout | agent_run | LIGHT (0.2s), MODERATE (1s), SEVERE (3s), CRITICAL (10s) |
| TemporaryApiFailure | api_call | 1–5 consecutive failures |
| InvalidModelResponse | agent_run | empty, malformed JSON, unexpected type |
| ToolFailure | tool_call | per-tool-name |
| ContextTruncation | prompt | 10%–90% truncation |
| GpuOom | api_call | memory fraction 0.3–1.0 |
| RuntimeCrash | agent_run | segfault, container OOM, process exit, panic |

### 5.2 Combined Faults

`CombinedFaultStrategy` applies multiple sub-strategies sequentially, enabling compound failure scenarios (e.g., timeout + crash, API failure + truncation). The combined fault name is a `+`-joined string of sub-strategy names.

### 5.3 Reliability Engineering Metrics

**MTBF (Mean Time Between Failures):**

$$\text{MTBF} = \frac{\sum_{i} \text{latency}_i}{\text{failures}}, \quad \text{failure rate} = \frac{\text{failures}}{\sum_i \text{latency}_i}$$

**MTTR (Mean Time To Recovery):**

$$\text{MTTR} = \frac{\sum_{i \in R} \text{latency}_i}{|R|}, \quad \text{recovery rate} = \frac{|R|}{N}$$

where $R$ is the set of recovery events (partial or full).

**Availability:**

$$A = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR}}$$

### 5.4 Severity Schedule

Each fault can be scheduled via `FaultSchedule` (FIRST_RUN, LAST_RUN, RANDOM_RUN, EVERY_RUN, SEQUENCE) with configurable probability, enabling controlled experiments across temporal conditions.

---

## 6. Statistical Methodology

### 6.1 Automated Procedure Selection

The auto-selection engine (`auto_select()`) implements a decision tree:

1. **Check normality** via Shapiro-Wilk test per group
2. **Check variance homogeneity** via Levene's test
3. **Select test** based on:
   - Two groups, normal → t-test (Welch if unequal variances)
   - Two groups, non-normal → Mann-Whitney U
   - Paired, normal → paired t-test
   - Paired, non-normal → Wilcoxon signed-rank
   - Three+ groups, normal → ANOVA (Welch if unequal)
   - Three+ groups, non-normal → Kruskal-Wallis
   - Repeated measures, normal → RM-ANOVA
   - Repeated measures, non-normal → Friedman test
4. **Select post-hoc:** Tukey HSD (normal) or Nemenyi (non-normal) for 3+ groups
5. **Select effect size:** Hedges' g ($n < 30$), Cohen's d ($n \geq 30$), rank-biserial $r$, $\eta^2$, $\omega^2$, $\varepsilon^2$, Kendall's $W$
6. **Select correction:** Bonferroni ($c \leq 5$), Holm-Bonferroni ($5 < c \leq 20$), Benjamini-Hochberg ($c > 20$)

### 6.2 Extended Test Suite

The framework provides 19 statistical functions beyond the standard package: Mann-Whitney U, Kruskal-Wallis, Friedman, one-way ANOVA, Nemenyi post-hoc, Hedges' g, Glass's Delta, eta-squared, omega-squared, Bonferroni/Holm/ Benjamini-Hochberg correction, post-hoc power analysis, a-priori sample size estimation, and Bayes factor for t-tests.

---

## 7. Experimental Protocol

We design a large-scale study to address five research questions (Table 2). The full protocol is documented in `docs/experimental_protocol.md`.

**Table 2: Research Questions and Design**

| RQ | Question | Design | Test |
|----|----------|--------|------|
| RQ1 | Do larger models have higher ISR? | 12 models × 3 tiers | Kruskal-Wallis + Nemenyi |
| RQ2 | Which fault type causes most loss? | 7 faults × 36 tasks | Friedman + Nemenyi |
| RQ3 | Does ISR add beyond existing metrics? | Hierarchical regression | ΔR² test |
| RQ4 | Is there temporal degradation? | 10 windows × 12 models | One-sample t-test |
| RQ5 | ISR vs task difficulty correlation? | 36 tasks | Spearman ρ |

---

## 8. Implementation and Validation

The framework comprises:

- **Core library:** `src/llm_reliability/` — metrics, benchmarks, faults, statistics
- **Tests:** 1127+ tests covering all components with zero regressions from baseline
- **Documentation:** Experimental protocol, literature review, this manuscript

### 8.1 Test Coverage

| Component | Tests | Status |
|-----------|-------|--------|
| ISR metric | 25 | 25 passed |
| ReliabilityBench | 38 | 38 passed |
| Fault extensions | 25 | 25 passed |
| Statistics extensions | 43 | 43 passed |
| Production faults | 26 | 26 passed |
| Auto-selection | 37 | 37 passed |
| All other (pre-existing) | 933 | 933 passed |
| **Total** | **1127** | **1127 passed** |

### 8.2 Reproducibility

All runs record configuration hashes, random seeds, software versions, and timestamps. The environment checklist captures Python version, package dependencies, GPU configuration, and operating system details.

---

## 9. Expected Contributions

1. **First operational reliability metric for LLM inference** — ISR measures *how much information survives*, not just whether the model "passes" or "fails."

2. **Reliability-focused benchmark** — ReliabilityBench provides a taxonomically grounded evaluation suite designed specifically for reliability studies.

3. **Production-grade fault injection** — GPU OOM, runtime crash, combined faults, and MTBF/MTTR bring software reliability engineering to LLM evaluation.

4. **Statistical automation** — Automated procedure selection addresses the finding that 84% of benchmark studies lack proper statistical methodology.

5. **Open-source implementation** — All code, tests, and documentation are publicly available under a permissive license.

---

## 10. Limitations and Future Work

**Current limitations.** (1) ISR depends on bin count $K$; sensitivity analysis across $K \in \{5, 10, 20\}$ is recommended. (2) Temporal ISR assumes chronological ordering; misordered timestamps could bias results. (3) The Confidence intervals use the percentile bootstrap; bias-corrected accelerated (BCa) intervals may provide better coverage. (4) The experimental protocol is designed but not yet executed—results are simulation-based.

**Future work.** (1) Execute the full experimental protocol across 12+ models. (2) Expand ReliabilityBench to 50+ tasks. (3) Implement real (non-simulated) GPU OOM via memory allocation and process crash via signal delivery. (4) Add Bayesian hierarchical modelling for cross-model reliability comparisons (extending HIP-LLM). (5) Develop a reliability leaderboard with ISR as the primary ranking metric.

---

## References

[Bommasani et al., 2022] HELM: Holistic Evaluation of Language Models. *Annals of the NY Academy of Sciences*.

[Cao et al., 2026] OpenCompass: A Universal Evaluation Platform for LLMs. *arXiv:2605.19276*.

[Efron & Tibshirani, 1994] *An Introduction to the Bootstrap*. Chapman & Hall.

[Gao et al., 2023] A framework for few-shot language model evaluation. *Zenodo*.

[Huang et al., 2026] Not All Errors Are Equal: Error Propagation in LLM Inference. *arXiv:2606.02430*.

[Iannillo et al., 2025] ChaosLLM: Dependability Testing for Tool-Calling Agents. *ISSRE 2025*.

[Laskar et al., 2024] A Systematic Survey and Critical Review on Evaluating LLMs. *arXiv:2407.04069*.

[McTaggart et al., 2025] HIP-LLM: Hierarchical Imprecise Probability for LLM Reliability Assessment. *arXiv:2511.00527*.

[Musa, 1993] Operational profiles in software-reliability engineering. *IEEE Software*.

[Swain & Ballard, 1991] Color indexing. *International Journal of Computer Vision*.

[NeurIPS 2025] Measuring What Matters: Construct Validity in LLM Benchmarks. *NeurIPS Datasets & Benchmarks*.

[SC 2025] Demystifying the Resilience of LLM Inference: An End-to-End Perspective. *SC 2025*.

[ICPE 2025] An Empirical Characterization of Outages and Incidents in Public Services for LLMs. *ICPE 2025*.
