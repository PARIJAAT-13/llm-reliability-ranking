[35mdata/gaia_sample.json[m[36m:[m[32m3[m[36m:[m    "task_id": "[1;31mgaia[m_001",
[35mdata/gaia_sample.json[m[36m:[m[32m11[m[36m:[m    "task_id": "[1;31mgaia[m_002",
[35mdata/gaia_sample.json[m[36m:[m[32m19[m[36m:[m    "task_id": "[1;31mgaia[m_003",
[35mdata/gaia_sample.json[m[36m:[m[32m27[m[36m:[m    "task_id": "[1;31mgaia[m_004",
[35mdata/gaia_sample.json[m[36m:[m[32m35[m[36m:[m    "task_id": "[1;31mgaia[m_005",
[35mpaper/PreExecutionReport.md[m[36m:[m[32m39[m[36m:[m- **GAIA**: `data/cache/[1;31mgaia[m_validation.jsonl` — Cached & validated.
[35mscripts/run_experiment.py[m[36m:[m[32m65[m[36m:[mimport llm_reliability.benchmarks.adapters.[1;31mgaia[m_adapter         # noqa: F401
[35mscripts/run_large_scale_experiment.py[m[36m:[m[32m48[m[36m:[mimport llm_reliability.benchmarks.adapters.[1;31mgaia[m_adapter          # noqa: F401
[35msrc/llm_reliability/benchmarks/adapters/gaia_adapter.py[m[36m:[m[32m20[m[36m:[mfrom llm_reliability.benchmarks.adapters.[1;31mgaia[m_models import (
[35msrc/llm_reliability/benchmarks/adapters/gaia_adapter.py[m[36m:[m[32m24[m[36m:[mfrom llm_reliability.benchmarks.adapters.[1;31mgaia[m_utils import normalize_[1;31mgaia[m_answer
[35msrc/llm_reliability/benchmarks/adapters/gaia_adapter.py[m[36m:[m[32m109[m[36m:[m            software_versions={"[1;31mgaia[m": "1.0"},
[35msrc/llm_reliability/benchmarks/adapters/gaia_adapter.py[m[36m:[m[32m127[m[36m:[m            expected_norm = normalize_[1;31mgaia[m_answer(str(expected))
[35msrc/llm_reliability/benchmarks/adapters/gaia_adapter.py[m[36m:[m[32m128[m[36m:[m            output_norm = normalize_[1;31mgaia[m_answer(str(agent_output))
[35msrc/llm_reliability/benchmarks/adapters/gaia_utils.py[m[36m:[m[32m14[m[36m:[mdef normalize_[1;31mgaia[m_answer(answer: str) -> str:
[35msrc/llm_reliability/benchmarks/dataset_manager.py[m[36m:[m[32m33[m[36m:[m    "[1;31mgaia[m": {
[35msrc/llm_reliability/benchmarks/dataset_manager.py[m[36m:[m[32m37[m[36m:[m        "filename": "[1;31mgaia[m_validation.jsonl",
[35msrc/llm_reliability/orchestration/experiment_orchestrator.py[m[36m:[m[32m68[m[36m:[m    "GAIA": "data/[1;31mgaia[m.json",
[35msrc/llm_reliability/orchestration/experiment_orchestrator.py[m[36m:[m[32m69[m[36m:[m    "[1;31mgaia[m": "data/[1;31mgaia[m.json",
[35mtests/fixtures/sample_gaia_dataset.json[m[36m:[m[32m3[m[36m:[m    "task_id": "[1;31mgaia[m_t1",
[35mtests/fixtures/sample_gaia_dataset.json[m[36m:[m[32m11[m[36m:[m    "task_id": "[1;31mgaia[m_t2",
[35mtests/records/test_evaluation_record.py[m[36m:[m[32m43[m[36m:[m        benchmark="[1;31mgaia[m",
[35mtests/records/test_evaluation_record.py[m[36m:[m[32m57[m[36m:[m    assert evaluation.benchmark == "[1;31mgaia[m"
[35mtests/test_gaia_adapter.py[m[36m:[m[32m7[m[36m:[mfrom llm_reliability.benchmarks.adapters.[1;31mgaia[m_adapter import GAIAAdapter
[35mtests/test_gaia_adapter.py[m[36m:[m[32m8[m[36m:[mfrom llm_reliability.benchmarks.adapters.[1;31mgaia[m_utils import normalize_[1;31mgaia[m_answer
[35mtests/test_gaia_adapter.py[m[36m:[m[32m45[m[36m:[mdef test_[1;31mgaia[m_adapter_loading_and_retrieval(config):
[35mtests/test_gaia_adapter.py[m[36m:[m[32m56[m[36m:[mdef test_[1;31mgaia[m_execution_and_evaluation(config):
[35mtests/test_gaia_adapter.py[m[36m:[m[32m75[m[36m:[mdef test_[1;31mgaia[m_utils_normalization():
[35mtests/test_gaia_adapter.py[m[36m:[m[32m76[m[36m:[m    assert normalize_[1;31mgaia[m_answer("Paris.") == "paris"
[35mtests/test_gaia_adapter.py[m[36m:[m[32m77[m[36m:[m    assert normalize_[1;31mgaia[m_answer(" Paris ") == "paris"
[35mtests/test_gaia_adapter.py[m[36m:[m[32m78[m[36m:[m    assert normalize_[1;31mgaia[m_answer("PARIS!") == "paris"
[35mtests/test_gaia_adapter.py[m[36m:[m[32m79[m[36m:[m    assert normalize_[1;31mgaia[m_answer(None) == ""  # type: ignore
[35mtests/test_gaia_adapter.py[m[36m:[m[32m82[m[36m:[mdef test_[1;31mgaia[m_invalid_dataset_schema(tmp_path):
[35mtests/test_gaia_adapter.py[m[36m:[m[32m103[m[36m:[mdef test_[1;31mgaia[m_duplicate_ids(tmp_path):
[35mtests/test_gaia_adapter.py[m[36m:[m[32m143[m[36m:[mdef test_[1;31mgaia[m_metadata_and_determinism(config):
[35mtests/test_gaia_adapter.py[m[36m:[m[32m163[m[36m:[mdef test_[1;31mgaia[m_missing_dataset(tmp_path):
