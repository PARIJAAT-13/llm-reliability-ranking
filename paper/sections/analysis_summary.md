# LLM Reliability Ranking Study: Analysis Summary
Generated: 2026-07-25 02:57:25

## 1. Experimental Overview

- **Total Evaluations**: 31,981
- **Overall Success Rate**: 0.8901
- **Fault Injection Rate**: 0.3039
- **Unique Agents**: 21
- **Unique Benchmarks**: 4
- **Number of Experiments**: 142

## 2. Final Rankings (Borda Consensus)

| Rank | Agent | Borda Score |
|------|-------|-------------|
| 1 | 9b | 1.0000 |
| 2 | 8b | 0.6875 |
| 3 | 7b | 0.5000 |
| 4 | 7b | 0.1667 |
| 5 | claude-3-5-sonnet | 0.1250 |
| 6 | llama-3.3-70b | 0.1250 |
| 7 | gemini-1.5-pro | 0.1250 |
| 8 | qwen-2.5-72b | 0.1250 |
| 9 | gpt-4o | 0.1250 |
| 10 | deepseek-chat | 0.1250 |
| 11 | mini | 0.1250 |
| 12 | mock | 0.0417 |
| 13 | OfflineLlamaAgent | 0.0417 |
| 14 | latest | 0.0208 |
| 15 | MockAgent | 0.0000 |
| 16 | mock_agent | 0.0000 |
| 17 | OfflineQwenAgent | 0.0000 |

## 3. Agent Reliability Summary

| Agent | Composite Rel. | Success Rate | Fault Tol. | Perturb. Rob. |
|-------|----------------|--------------|------------|----------------|
| OfflineLlamaAgent | 1.00±0.00 | 1.00±0.00 | --- | --- |
| OfflineQwenAgent | 1.00±0.00 | 1.00±0.00 | --- | --- |
| claude-3-5-sonnet | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00 |
| deepseek | 1.00±0.00 | 0.00±0.00 | --- | --- |
| deepseek-chat | 1.00±0.00 | 0.75±0.43 | 1.00±0.00 | 1.00 |
| google | 1.00±0.00 | 0.00±0.00 | --- | --- |
| gemini-1.5-pro | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00 |
| llama | 0.98±0.06 | 0.38±0.47 | --- | --- |
| llama-3.3-70b | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00 |
| mock | 1.00±0.00 | 1.00±0.00 | --- | --- |
| mock_agent | 1.00±0.00 | 1.00±0.00 | --- | --- |
| 9b | 1.00±0.00 | 0.86±0.35 | --- | --- |
| 8b | 1.00±0.00 | 0.57±0.49 | --- | --- |
| 7b | 1.00±0.00 | 0.75±0.43 | --- | --- |
| mini | 1.00±0.00 | 0.50±0.50 | --- | --- |
| 7b | 1.00±0.00 | 0.88±0.33 | --- | --- |
| latest | 0.90±0.10 | 0.90±0.10 | --- | --- |
| openai | 1.00±0.00 | 0.00±0.00 | --- | --- |
| gpt-4o | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00 |
| qwen-2.5-72b | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00 |

## 4. Fault Injection Recovery Analysis

| Fault Type | Count | Recovery Rate | Avg Retries |
|------------|-------|---------------|-------------|
| artificial_timeout | 1620 | 100.00% | 0.0 |
| context_truncation | 1620 | 100.00% | 0.0 |
| invalid_model_response | 1620 | 100.00% | 0.0 |
| network_interruption | 1620 | 0.00% | 3.0 |
| temporary_api_failure | 1620 | 100.00% | 1.0 |
| tool_failure | 1620 | 0.00% | 3.0 |

## 5. Ranking Consistency

### success

| Agent | Mean Rank | Std Dev | Min | Max |
|-------|-----------|---------|-----|-----|
| 9b | 1.00 | 0.00 | 1 | 1 |
| claude-3-5-sonnet | 1.00 | 0.00 | 1 | 1 |
| llama-3.3-70b | 1.00 | 0.00 | 1 | 1 |
| gemini-1.5-pro | 1.00 | 0.00 | 1 | 1 |
| qwen-2.5-72b | 1.00 | 0.00 | 1 | 1 |
| gpt-4o | 1.00 | 0.00 | 1 | 1 |
| deepseek-chat | 1.00 | 0.00 | 1 | 1 |
| mock | 1.00 | 0.00 | 1 | 1 |
| OfflineLlamaAgent | 1.00 | 0.00 | 1 | 1 |
| MockAgent | 2.00 | 0.00 | 2 | 2 |
| mock_agent | 2.00 | 0.00 | 2 | 2 |
| OfflineQwenAgent | 2.00 | 0.00 | 2 | 2 |
| 8b | 2.29 | 0.70 | 2 | 4 |
| 7b | 2.75 | 0.43 | 2 | 3 |
| 7b | 3.50 | 1.12 | 1 | 5 |
| mini | 5.00 | 1.00 | 4 | 6 |
| latest | 5.50 | 0.50 | 5 | 6 |

### reliability

| Agent | Mean Rank | Std Dev | Min | Max |
|-------|-----------|---------|-----|-----|
| 9b | 1.00 | 0.00 | 1 | 1 |
| claude-3-5-sonnet | 1.00 | 0.00 | 1 | 1 |
| llama-3.3-70b | 1.00 | 0.00 | 1 | 1 |
| gemini-1.5-pro | 1.00 | 0.00 | 1 | 1 |
| qwen-2.5-72b | 1.00 | 0.00 | 1 | 1 |
| gpt-4o | 1.00 | 0.00 | 1 | 1 |
| deepseek-chat | 1.00 | 0.00 | 1 | 1 |
| mock | 1.00 | 0.00 | 1 | 1 |
| OfflineLlamaAgent | 1.00 | 0.00 | 1 | 1 |
| 8b | 1.86 | 0.35 | 1 | 2 |
| MockAgent | 2.00 | 0.00 | 2 | 2 |
| mock_agent | 2.00 | 0.00 | 2 | 2 |
| OfflineQwenAgent | 2.00 | 0.00 | 2 | 2 |
| 7b | 2.75 | 0.43 | 2 | 3 |
| 7b | 4.00 | 0.71 | 3 | 5 |
| mini | 4.00 | 0.00 | 4 | 4 |
| latest | 6.00 | 0.00 | 6 | 6 |

## 6. Perturbation Robustness

- **formatting**: 100.00% success rate (n=1620)
- **prompt_wrapper**: 100.00% success rate (n=1620)
- **reordering**: 100.00% success rate (n=1620)
- **synonym**: 100.00% success rate (n=1620)
- **whitespace**: 100.00% success rate (n=1620)

## 7. Task Difficulty Impact

- **1**: 14.29% success rate (n=7)
- **2**: 0.00% success rate (n=28)
- **easy**: 89.03% success rate (n=16030)
- **hard**: 89.55% success rate (n=15656)
- **medium**: 66.92% success rate (n=260)
