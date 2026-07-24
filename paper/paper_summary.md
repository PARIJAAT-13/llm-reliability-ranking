# LLM Reliability Ranking — Comprehensive Experimental Findings

- **Benchmarks Evaluated**: 12 (GAIA, MMLU, HellaSwag, HumanEval, MBPP, TruthfulQA, GSM8K, ARC, Winogrande, PIQA, AgentBoard, SWE-bench Lite)
- **Runtimes Evaluated**: 4 (Ollama, llama.cpp, vLLM, HuggingFace Transformers)
- **Models Evaluated**: 6
- **Random Seeds**: 5 (42, 100, 2026, 777, 999)
- **Ranking Overlap**: 86.67%
- **Ranking Divergence**: 13.33%
- **Mean Rank Displacement**: 0.67 positions

## Generated Publication Artifacts
- Figures: `paper/figures/fig1_ranking_bump_chart.pdf`, `fig2_success_vs_reliability_scatter.pdf`
- LaTeX Table: `paper/tables/table1_agent_reliability_matrix.tex`
