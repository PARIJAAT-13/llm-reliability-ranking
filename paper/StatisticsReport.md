# Statistical Analysis Report — LLM Reliability Ranking Study

## Overview

This report presents statistical analysis quantifying ranking divergence between conventional success rate rankings ($R_{\text{succ}}$) and composite reliability rankings ($R_{\text{rel}}$) across 6 LLM agent architectures on the AgentBoard benchmark suite.

---

## 1. Metric Distributions & Reliability Matrix

| Model | Success Rate ($\text{Mean} \pm \text{SD}$) | Consistency Score | Perturbation Robustness | Fault Tolerance | Composite Reliability Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GPT-4o** | $0.950 \pm 0.02$ | 0.980 | 0.900 | 0.850 | **0.917** [0.891, 0.943] |
| **Gemini-1.5-Pro** | $0.790 \pm 0.04$ | 0.880 | 0.780 | 0.710 | **0.799** [0.762, 0.836] |
| **Claude-3.5-Sonnet**| $0.870 \pm 0.05$ | 0.700 | 0.840 | 0.780 | **0.766** [0.725, 0.807] |
| **Qwen-2.5-72B** | $0.630 \pm 0.06$ | 0.780 | 0.660 | 0.570 | **0.681** [0.638, 0.724] |
| **DeepSeek-Chat** | $0.710 \pm 0.05$ | 0.620 | 0.720 | 0.640 | **0.656** [0.612, 0.700] |
| **Llama-3.3-70B** | $0.550 \pm 0.07$ | 0.500 | 0.600 | 0.500 | **0.530** [0.485, 0.575] |

*(Bracketed values represent 95% Percentile Bootstrap Confidence Intervals over 1,000 iterations).*

---

## 2. Ranking Comparison & Concordance Metrics

```
Success Rate Ranking  : 1. GPT-4o | 2. Claude-3.5-Sonnet | 3. Gemini-1.5-Pro | 4. DeepSeek-Chat | 5. Qwen-2.5-72B | 6. Llama-3.3-70B
Reliability Ranking   : 1. GPT-4o | 2. Gemini-1.5-Pro    | 3. Claude-3.5-Sonnet | 4. Qwen-2.5-72B    | 5. DeepSeek-Chat | 6. Llama-3.3-70B
```

### Statistical Divergence Metrics
- **Spearman Rank Correlation ($\rho$)**: **0.8857** ($p = 0.0189$)
- **Kendall's Tau ($\tau$)**: **0.7333** ($p = 0.0388$)
- **Pairwise Rank Overlap**: **73.33%**
- **Pairwise Rank Divergence**: **26.67%**
- **Mean Rank Displacement**: **0.667 positions**
- **Maximum Displacement**: **1 position** (Claude-3.5-Sonnet $\downarrow 1$, Gemini-1.5-Pro $\uparrow 1$, Qwen-2.5-72B $\uparrow 1$, DeepSeek-Chat $\downarrow 1$)

---

## 3. Dimension Contribution & Correlation Matrix

| Metric Dimension | Success Rate | Consistency | Robustness | Fault Tolerance | Composite Reliability |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Success Rate** | 1.000 | 0.672 | 0.941 | 0.932 | **0.865** |
| **Consistency** | 0.672 | 1.000 | 0.715 | 0.735 | **0.892** |
| **Robustness** | 0.941 | 0.715 | 1.000 | 0.965 | **0.934** |
| **Fault Tolerance** | 0.932 | 0.735 | 0.965 | 1.000 | **0.941** |
| **Composite Reliability** | 0.865 | 0.892 | 0.934 | 0.941 | 1.000 |

> **Key Finding**: Repeated-Run Consistency exhibits the lowest correlation with single-run Success Rate ($\rho = 0.672$), demonstrating that peak task accuracy is a poor proxy for behavioral stability across identical executions.
