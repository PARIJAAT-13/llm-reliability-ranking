"""Tests for model registry, ModelInfo validation, and discovery."""

import pytest

from llm_reliability.models import (SUPPORTED_OLLAMA_MODELS,
                                    DuplicateModelError, ModelInfo,
                                    ModelNotFoundError, ModelRegistry,
                                    discover_local_models, merge_discovered,
                                    populate_registry)
from llm_reliability.models.discovery import _infer_family


@pytest.fixture(autouse=True)
def reset_registry():
    ModelRegistry.clear()
    yield
    ModelRegistry.clear()


def _make_model(**kwargs) -> ModelInfo:
    defaults = dict(
        family="Test",
        name="Test Model 7B",
        parameters="7B",
        parameter_count=7.0,
        ollama_identifier="test:7b",
    )
    defaults.update(kwargs)
    return ModelInfo(**defaults)


# --------------------------------------------------------------------------- #
# ModelInfo validation
# --------------------------------------------------------------------------- #


def test_model_info_valid():
    m = _make_model()
    assert m.family == "Test"
    assert m.ollama_identifier == "test:7b"
    assert m.provider == "ollama"
    assert m.runtime == "ollama"
    assert m.status == "supported"


def test_model_info_missing_colon_in_identifier():
    with pytest.raises(ValueError, match="must contain a colon"):
        ModelInfo(
            family="Test",
            name="Test",
            parameters="7B",
            parameter_count=7.0,
            ollama_identifier="invalid",
        )


def test_model_info_empty_identifier():
    with pytest.raises(ValueError):
        ModelInfo(
            family="Test",
            name="Test",
            parameters="7B",
            parameter_count=7.0,
            ollama_identifier="",
        )


def test_model_info_negative_parameter_count():
    with pytest.raises(ValueError):
        _make_model(parameter_count=-1.0)


def test_model_info_custom_provider():
    m = _make_model(provider="ollama", runtime="ollama")
    assert m.provider == "ollama"
    assert m.runtime == "ollama"


def test_model_info_metadata_default():
    m = _make_model()
    assert m.metadata == {}


def test_model_info_custom_metadata():
    m = _make_model(metadata={"key": "value", "source": "huggingface"})
    assert m.metadata["key"] == "value"
    assert m.metadata["source"] == "huggingface"


def test_model_info_experimental_status():
    m = _make_model(status="experimental")
    assert m.status == "experimental"


def test_model_info_context_window_default():
    m = _make_model()
    assert m.context_window == 0


def test_model_info_context_window_custom():
    m = _make_model(context_window=131072)
    assert m.context_window == 131072


def test_model_info_architecture_default():
    m = _make_model()
    assert m.architecture == "Transformer (Decoder-Only)"


# --------------------------------------------------------------------------- #
# Registry operations
# --------------------------------------------------------------------------- #


def test_registry_register_and_get():
    model = _make_model()
    ModelRegistry.register(model)
    assert ModelRegistry.exists("test:7b")
    retrieved = ModelRegistry.get("test:7b")
    assert retrieved.name == "Test Model 7B"


def test_registry_get_nonexistent():
    with pytest.raises(ModelNotFoundError):
        ModelRegistry.get("nonexistent:7b")


def test_registry_duplicate_detection():
    ModelRegistry.register(_make_model())
    with pytest.raises(DuplicateModelError, match="Duplicate model identifier"):
        ModelRegistry.register(_make_model())


def test_registry_register_many_with_duplicates():
    models = [
        _make_model(),
        _make_model(ollama_identifier="test2:7b"),
        _make_model(),
    ]
    with pytest.raises(DuplicateModelError, match="Duplicate model identifier"):
        ModelRegistry.register_many(models)


def test_registry_register_invalid_type():
    with pytest.raises(TypeError, match="Expected ModelInfo"):
        ModelRegistry.register("not_a_model")


def test_registry_list_identifiers():
    ModelRegistry.register(_make_model(ollama_identifier="b:7b"))
    ModelRegistry.register(_make_model(ollama_identifier="a:7b"))
    assert ModelRegistry.list_identifiers() == ["a:7b", "b:7b"]


def test_registry_list_families():
    ModelRegistry.register(_make_model(family="Beta", ollama_identifier="beta:7b"))
    ModelRegistry.register(_make_model(family="Alpha", ollama_identifier="alpha:7b"))
    assert ModelRegistry.list_families() == ["Alpha", "Beta"]


def test_registry_list_by_family():
    ModelRegistry.register(
        _make_model(family="F1", ollama_identifier="f1:a:7b", parameter_count=7.0)
    )
    ModelRegistry.register(_make_model(family="F2", ollama_identifier="f2:a:7b"))
    ModelRegistry.register(
        _make_model(family="F1", ollama_identifier="f1:b:3b", parameter_count=3.0)
    )
    by_family = ModelRegistry.list_by_family()
    assert "F1" in by_family
    assert "F2" in by_family
    assert len(by_family["F1"]) == 2
    assert by_family["F1"][0].parameter_count < by_family["F1"][1].parameter_count


def test_registry_count():
    assert ModelRegistry.count() == 0
    ModelRegistry.register(_make_model())
    assert ModelRegistry.count() == 1
    ModelRegistry.register(_make_model(ollama_identifier="another:7b"))
    assert ModelRegistry.count() == 2


def test_registry_unregister():
    ModelRegistry.register(_make_model())
    ModelRegistry.unregister("test:7b")
    assert not ModelRegistry.exists("test:7b")


def test_registry_unregister_nonexistent():
    with pytest.raises(ModelNotFoundError, match="not found"):
        ModelRegistry.unregister("ghost:7b")


def test_registry_to_dict_list():
    ModelRegistry.register(_make_model())
    dict_list = ModelRegistry.to_dict_list()
    assert len(dict_list) == 1
    assert dict_list[0]["ollama_identifier"] == "test:7b"
    assert dict_list[0]["family"] == "Test"


def test_registry_reset():
    ModelRegistry.register(_make_model())
    ModelRegistry.reset()
    assert ModelRegistry.count() == 0
    assert ModelRegistry._initialized is True


def test_registry_clear():
    ModelRegistry.register(_make_model())
    ModelRegistry.clear()
    assert ModelRegistry._initialized is False


def test_registry_initialize_twice():
    ModelRegistry.initialize()
    ModelRegistry.initialize()
    assert ModelRegistry._initialized is True


# --------------------------------------------------------------------------- #
# SUPPORTED_OLLAMA_MODELS catalogue
# --------------------------------------------------------------------------- #


def test_supported_models_populate():
    ModelRegistry.register_many(SUPPORTED_OLLAMA_MODELS)
    assert ModelRegistry.count() >= 40


def test_supported_models_all_have_colon():
    for model in SUPPORTED_OLLAMA_MODELS:
        assert ":" in model.ollama_identifier, f"{model.name} missing colon"


def test_supported_models_all_have_positive_params():
    for model in SUPPORTED_OLLAMA_MODELS:
        assert model.parameter_count > 0, f"{model.name} has parameter_count=0"


def test_supported_models_all_have_family():
    for model in SUPPORTED_OLLAMA_MODELS:
        assert model.family, f"{model.ollama_identifier} has empty family"


def test_supported_models_provider_is_ollama():
    for model in SUPPORTED_OLLAMA_MODELS:
        assert (
            model.provider == "ollama"
        ), f"{model.ollama_identifier} has provider={model.provider}"


def test_supported_models_all_unique_identifiers():
    identifiers = [m.ollama_identifier for m in SUPPORTED_OLLAMA_MODELS]
    duplicates = {x for x in identifiers if identifiers.count(x) > 1}
    assert not duplicates, f"Duplicate identifiers: {duplicates}"


def test_supported_models_multiple_families():
    families = {m.family for m in SUPPORTED_OLLAMA_MODELS}
    assert len(families) >= 10, f"Expected >=10 families, got {len(families)}"


def test_populate_registry():
    count = populate_registry()
    assert count >= 40
    assert ModelRegistry.exists("llama3.1:8b")
    assert ModelRegistry.exists("qwen2.5:7b")
    assert ModelRegistry.exists("mistral:7b")
    assert ModelRegistry.exists("gemma2:9b")
    assert ModelRegistry.exists("tinyllama:latest")


# --------------------------------------------------------------------------- #
# Discovery inference utilities
# --------------------------------------------------------------------------- #


def test_infer_family_llama():
    assert _infer_family("llama3.1:8b") == "Llama"


def test_infer_family_qwen():
    assert _infer_family("qwen2.5:7b") == "Qwen"


def test_infer_family_mistral():
    assert _infer_family("mistral:7b") == "Mistral"


def test_infer_family_gemma():
    assert _infer_family("gemma2:9b") == "Gemma"


def test_infer_family_phi():
    assert _infer_family("phi3:mini") == "Phi"


def test_infer_family_deepseek():
    assert _infer_family("deepseek-r1:7b") == "DeepSeek"


def test_infer_family_codellama():
    assert _infer_family("codellama:34b") == "Llama"


def test_infer_family_codegemma():
    assert _infer_family("codegemma:7b") == "Gemma"


def test_infer_family_starcoder2():
    assert _infer_family("starcoder2:15b") == "StarCoder2"


def test_infer_family_unknown():
    assert _infer_family("unknown-model:latest") == "Unknown-model"


def test_infer_family_dbrx():
    assert _infer_family("dbrx:132b") == "DBRX"


def test_infer_family_yi():
    assert _infer_family("yi-coder:9b") == "Yi"


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


def test_model_info_empty_name():
    m = _make_model(name="")
    assert m.name == ""


def test_model_info_quantization_field():
    m = _make_model(quantization="Q4_K_M")
    assert m.quantization == "Q4_K_M"


def test_model_info_none_quantization():
    m = _make_model()
    assert m.quantization is None


def test_model_info_ram_vram_defaults():
    m = _make_model()
    assert m.recommended_ram_gb is None
    assert m.recommended_vram_gb is None


def test_registry_list_models_sorted():
    ModelRegistry.register(_make_model(ollama_identifier="z:7b"))
    ModelRegistry.register(_make_model(ollama_identifier="a:7b"))
    models = ModelRegistry.list_models()
    assert models[0].ollama_identifier == "a:7b"
    assert models[1].ollama_identifier == "z:7b"


def test_populate_registry_idempotent():
    count1 = populate_registry()
    count2 = populate_registry()
    assert count1 == count2 > 0


def test_discovery_merge_no_duplicates(reset_registry):
    populate_registry()
    existing_count = ModelRegistry.count()
    merge_discovered([])
    assert ModelRegistry.count() == existing_count
