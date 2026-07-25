# Systematic Literature Review: LLM Reliability Evaluation

## 1. Executive Summary

This review surveys the state of LLM reliability evaluation through analysis of major benchmarking frameworks (HELM, lm-evaluation-harness, OpenCompass, DeepEval, LangSmith), recent conference proceedings (NeurIPS 2024-25, ICML 2024-25, ICPE 2025, ISSRE 2025), and reliability engineering literature adapted to AI systems. The central finding is that **no existing framework provides an operational reliability metric for LLM inference**—all current approaches evaluate capability (accuracy on static benchmarks) rather than reliability (probability of correct operation under real-world conditions including faults).

## 2. Major Evaluation Frameworks

### 2.1 HELM (Holistic Evaluation of Language Models)
**Source:** Bommasani et al., 2022; crfm.stanford.edu/helm
- 16 core scenarios across 7 metrics: accuracy, calibration, robustness, fairness, bias, toxicity, efficiency
- 30 models evaluated under standardized conditions
- **Limitation:** Robustness metric measures performance on perturbed inputs, not operational reliability under system faults
- **Gap:** No fault injection, no recovery measurement, no temporal reliability

### 2.2 lm-evaluation-harness (EleutherAI)
**Source:** github.com/EleutherAI/lm-evaluation-harness
- 60+ benchmarks, 500+ subtasks, industry standard for academic evaluation
- Powers HuggingFace Open LLM Leaderboard
- **Limitation:** Purely accuracy-based on static academic datasets
- **Gap:** No reliability dimension whatsoever; scores vary 4-20 points across harness implementations (SourceScore, 2026)

### 2.3 OpenCompass
**Source:** Cao et al., 2026; github.com/open-compass/opencompass
- 100+ datasets, ~400K questions, 20+ model profiles
- Strong multilingual support, modular architecture
- **Limitation:** Same accuracy-centric paradigm; adds breadth but not reliability depth
- **Gap:** No operational evaluation

### 2.4 DeepEval
**Source:** github.com/confident-ai/deepeval
- 50+ metrics, pytest-native, LLM-as-a-judge
- CI/CD integration, synthetic data generation
- **Limitation:** Focus on RAG/agent output quality, not system reliability
- **Gap:** No fault tolerance or resilience measurement

### 2.5 LangSmith
**Source:** docs.langchain.com/langsmith
- Observability, dataset management, production monitoring
- Online/offline evaluation split
- **Limitation:** Evaluation framework, not reliability measurement framework
- **Gap:** No reliability metric definitions

## 3. Key Research Papers

### 3.1 "Large Language Model Benchmarks Do Not Test Reliability"
**Venue:** OpenReview 2025
**Finding:** Current "saturated" benchmarks contain 5%+ label errors. Frontier models still fail on elementary math when benchmark noise is removed. Proposed "platinum benchmarks" with curated data.
**Key Insight:** Reliability is distinct from capability—models can solve PhD-level problems while failing basic tasks.

### 3.2 "HIP-LLM: Hierarchical Imprecise Probability for LLM Reliability"
**Venue:** arXiv 2025 (McTaggart et al.)
**Contribution:** First formal definition of LLM reliability as "probability of failure-free operation over specified future tasks under given Operational Profile." Hierarchical Bayesian model with imprecise priors.
**Five Research Gaps Identified:**
1. Static benchmarks vs dynamic operational use
2. Single accuracy point estimates vs probabilistic characterization
3. Independent task evaluation vs hierarchical dependencies
4. Failure probability vs probability of future failure-free runs
5. Point priors vs uncertainty over priors themselves

### 3.3 "Measuring What Matters: Construct Validity in LLM Benchmarks"
**Venue:** NeurIPS 2025 Datasets & Benchmarks
**Finding:** Systematic review of 445 benchmarks. Only 53.4% presented evidence of construct validity. Only 16% used statistical tests to compare models.
**Eight Recommendations:** Include statistical methods, error analysis, construct validity evidence.

### 3.4 "Enhancing Reliability in AI Inference Services"
**Venue:** arXiv 2025 (Ranganathan et al.)
**Finding:** Provider-internal analysis of 156 high-severity incidents. Taxonomy: ~60% inference engine failures, ~40% timeouts. ~74% auto-detected, ~28% required hotfix.
**Key Metrics:** TTD (time to detection), TTE (time to diagnosis), TTM (time to mitigation)

### 3.5 "An Empirical Characterization of Outages and Incidents in LLM Services"
**Venue:** ICPE 2025
**Finding:** MTTR/MTBF analysis across OpenAI, Anthropic, DeepSeek. ChatGPT has slowest recovery (median 1.07h operator, 1.23h user). Failures peak weekdays 8:00-16:00.

### 3.6 "Not All Errors Are Equal: Error Propagation in LLM Inference"
**Venue:** arXiv 2026 (Huang et al.)
**Contribution:** LLMFI fault injection framework for LLM inference stages. 17 takeaways on error propagation. Early decode iterations are most vulnerable.

### 3.7 "ChaosLLM: Dependability Testing for Tool-Calling Agents"
**Venue:** ISSRE 2025 (Iannillo et al.)
**Contribution:** Fault injection framework for LangChain agents. Failure modes: unreachable, slow response, hang, incorrect response. Task Success Rate (TSR) metric.

### 3.8 "Training Overhead Ratio: A Reliability Metric for LLM Training"
**Venue:** arXiv 2024
**Contribution:** TOR = optimal training time / observed training time. First reliability metric for LLM training systems. Fail-stop and fail-slow failure models.

### 3.9 "Demystifying the Resilience of LLM Inference: An End-to-End Perspective"
**Venue:** SC 2025
**Finding:** LLMs are NOT truly resilient to bit-flips. Aggregate accuracy masks significant text quality degradation. Generative tasks 2x more vulnerable than multiple-choice.

## 4. Research Gaps Identified

| Gap | Description | Existing Work | Our Position |
|-----|-------------|---------------|--------------|
| **G1** | No operational reliability metric for inference | Existing metrics are accuracy-based (HELM, lm-eval) | ISR metric fills this |
| **G2** | Static evaluation doesn't capture reliability | All frameworks evaluate on static datasets | ReliabilityBench with fault injection |
| **G3** | Fault injection for inference is nascent | LLMFI (2026), ChaosLLM (2025) only | Fault injection extensions + recovery metrics |
| **G4** | No reliability-focused dataset | All datasets target capability | ReliabilityBench (6 categories, 30 tasks) |
| **G5** | Statistical rigor lacking | Only 16% use stats (NeurIPS 2025) | Extended statistical suite |
| **G6** | No unified reliability taxonomy | Different definitions across papers | Formal ISR taxonomy |
| **G7** | Production studies are provider-specific | Provider-internal (OpenAI, Anthropic) | Generalizable fault injection framework |
| **G8** | No standard operational profile | HIP-LLM proposes but no standard | Task taxonomy in ReliabilityBench |

## 5. Our Contribution Positioning

The framework now implements:

1. **ISR (Information Survival Rate)** — First information-theoretic reliability metric for LLM inference. Measures fraction of information preserved under fault conditions. Addresses G1.

2. **ReliabilityBench** — 30-task taxonomy-driven dataset with fault/perturbation applicability. Addresses G4.

3. **Fault injection extensions** — Severity, scheduling, recovery metrics. Addresses G3.

4. **Extended statistical suite** — 19 new functions including Bayesian analysis, power analysis, multiple comparisons. Addresses G5.

## 6. Next Research Steps

1. **Stage 2:** Refine ISR mathematical formulation with formal proofs
2. **Stage 3:** Expand ReliabilityBench to 50+ tasks across more categories
3. **Stage 4:** Design and execute large-scale empirical study across 10+ models
4. **Stage 5:** Implement operational fault injection (OOM, crash, timeout)
5. **Stage 6:** Statistical validation with power analysis and effect sizes
6. **Stage 7:** Write complete manuscript
7. **Stage 8:** Prepare artifact evaluation package

## References

[Bommasani et al., 2022] HELM: Holistic Evaluation of Language Models. Annals of the NY Academy of Sciences.

[Cao et al., 2026] OpenCompass: A Universal Evaluation Platform for LLMs. arXiv:2605.19276.

[Huang et al., 2026] Not All Errors Are Equal: Error Propagation in LLM Inference. arXiv:2606.02430.

[Iannillo et al., 2025] ChaosLLM: Dependability Testing for Tool-Calling Agents. ISSRE 2025.

[McTaggart et al., 2025] HIP-LLM: Hierarchical Imprecise Probability for LLM Reliability Assessment. arXiv:2511.00527.

[Ranganathan et al., 2025] Enhancing Reliability in AI Inference Services. arXiv:2511.07424.

[Liang et al., 2024] Training Overhead Ratio: A Reliability Metric for LLM Training. arXiv:2408.07482.

[Laskar et al., 2024] A Systematic Survey and Critical Review on Evaluating LLMs. arXiv:2407.04069.

[NeurIPS 2025] Measuring What Matters: Construct Validity in LLM Benchmarks. NeurIPS 2025 Datasets & Benchmarks.

[ICPE 2025] An Empirical Characterization of Outages in LLM Services. ICPE 2025.

[SC 2025] Demystifying the Resilience of LLM Inference. SC 2025.
