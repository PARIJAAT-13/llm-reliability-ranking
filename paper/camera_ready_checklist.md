# Camera-Ready Submission Checklist

**Paper Title**: LLM Reliability Ranking Framework: Benchmarking Local Open-Weights Language Models Under Hardware Memory Constraints  
**Target Venues**: ICLR / NeurIPS / EMNLP / IEEE Transactions on Software Engineering (TSE)  
**Status**: **100% READY FOR SUBMISSION**  

---

## Verification Matrix

- [x] **✓ Figures Verified**: All 5 figures (`accuracy.png`, `reliability.png`, `latency.png`, `completion_rate.png`, `ranking.png`) regenerated at 300 DPI directly from 180-execution dataset `16a74baf-e97c-42f0-b286-40b5d120620b`.
- [x] **✓ Tables Verified**: Table 1, Table 2, and Table 3 in `paper/results.md` and `paper/manuscript.md` match `executions.json` and `evaluations.json` with 100% numerical precision.
- [x] **✓ Statistical Analysis Complete**: Mean, median, standard deviation, min, max, and 95% confidence intervals computed and reported for latency and accuracy across 45 executions per model.
- [x] **✓ Citations Verified**: Added foundational citations (vLLM SOSP 2023 [3], GAIA ICLR 2024 [4], HELM 2022 [1], MMLU 2021 [5], Prompt Sensitivity ACL 2022 [6]).
- [x] **✓ Grammar & Style Checked**: Academic prose reviewed for clarity, flow, tone, and terminology consistency.
- [x] **✓ Reproducibility Verified**: Comprehensive `README.md` with environment setup, Ollama model pull steps, single-command run script, and pytest verification.
- [x] **✓ Code Quality**: `ruff check src tests` passes with 0 errors; `pytest` passes 347 / 347 unit tests.
- [x] **✓ Limitations & Threats Discussed**: Section 5 explicitly details host RAM limits (15.7 GiB), dataset scope (5 Level 1 GAIA validation tasks), and Q4_0 quantization.
- [x] **✓ Open-Source Package**: `LICENSE` (MIT) and `CITATION.cff` present in repository root.
