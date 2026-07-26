"""Extended tests for SerializableModel — 40+ tests."""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_reliability.utils.serialization import SerializableModel

SHA256_HEX_LEN = 64


class SimpleModel(SerializableModel):
    name: str
    value: int = 0
    optional: str | None = None


class NestedModel(SerializableModel):
    inner: SimpleModel
    tag: str = "default"


class ModelWithList(SerializableModel):
    items: list[int] = []
    label: str = ""


class ModelWithDict(SerializableModel):
    mapping: dict[str, Any] = {}


class ModelWithOptionalList(SerializableModel):
    names: list[str] | None = None
    active: bool = True


class ModelWithNestedOptional(SerializableModel):
    data: dict[str, Any] | None = None
    inner: SimpleModel | None = None


class ModelWithAllTypes(SerializableModel):
    int_val: int = 0
    float_val: float = 0.0
    str_val: str = ""
    bool_val: bool = False
    list_val: list[int] = []
    dict_val: dict[str, int] = {}
    tuple_val: tuple[str, ...] = ()
    none_val: None = None


class TestSerializableModelConstruction:
    def test_simple_construction(self):
        m = SimpleModel(name="test", value=42)
        assert m.name == "test"
        assert m.value == 42
        assert m.optional is None

    def test_with_optional_set(self):
        m = SimpleModel(name="t", value=1, optional="present")
        assert m.optional == "present"

    def test_with_default_value(self):
        m = SimpleModel(name="t")
        assert m.value == 0

    def test_rejects_extra_field(self):
        with pytest.raises((TypeError, ValueError)):
            SimpleModel(name="t", unknown=True)

    def test_rejects_wrong_type(self):
        with pytest.raises((TypeError, ValueError)):
            SimpleModel(name=123, value="abc")

    def test_nested_model(self):
        inner = SimpleModel(name="inner", value=99)
        m = NestedModel(inner=inner)
        assert m.inner.name == "inner"
        assert m.inner.value == 99

    def test_list_field_default(self):
        m = ModelWithList(label="test")
        assert m.items == []

    def test_dict_field_default(self):
        m = ModelWithDict()
        assert m.mapping == {}

    def test_optional_list_none(self):
        m = ModelWithOptionalList()
        assert m.names is None

    def test_optional_list_provided(self):
        m = ModelWithOptionalList(names=["a", "b"])
        assert m.names == ["a", "b"]

    def test_all_types_defaults(self):
        m = ModelWithAllTypes()
        assert m.int_val == 0
        assert m.float_val == 0.0
        assert m.str_val == ""
        assert m.bool_val is False
        assert m.list_val == []
        assert m.dict_val == {}
        assert m.tuple_val == ()
        assert m.none_val is None

    def test_all_types_set(self):
        m = ModelWithAllTypes(
            int_val=42,
            float_val=3.14,
            str_val="hello",
            bool_val=True,
            list_val=[1, 2, 3],
            dict_val={"a": 1},
            tuple_val=("x", "y"),
        )
        assert m.int_val == 42
        assert m.float_val == 3.14
        assert m.str_val == "hello"
        assert m.bool_val is True
        assert m.list_val == [1, 2, 3]
        assert m.dict_val == {"a": 1}
        assert m.tuple_val == ("x", "y")

    def test_immutable(self):
        m = SimpleModel(name="t")
        with pytest.raises((TypeError, ValueError)):
            m.name = "changed"

    def test_nested_optional_set(self):
        inner = SimpleModel(name="inner")
        m = ModelWithNestedOptional(inner=inner)
        assert m.inner is not None
        assert m.inner.name == "inner"

    def test_nested_optional_none(self):
        m = ModelWithNestedOptional()
        assert m.inner is None


class TestCanonicalJSON:
    def test_simple_canonical_json(self):
        m = SimpleModel(name="test", value=42)
        parsed = json.loads(m.canonical_json())
        assert parsed["name"] == "test"
        assert parsed["value"] == 42

    def test_keys_sorted(self):
        m = SimpleModel(name="b", value=1)
        json_str = m.canonical_json()
        assert json_str == '{"name":"b","value":1}'

    def test_excludes_none(self):
        m = SimpleModel(name="t", value=1)
        assert "optional" not in m.canonical_json()

    def test_includes_false_bool(self):
        m = ModelWithOptionalList(active=False)
        assert '"active":false' in m.canonical_json()

    def test_includes_empty_list(self):
        m = ModelWithList(items=[])
        assert '"items":[]' in m.canonical_json()

    def test_includes_empty_dict(self):
        m = ModelWithDict(mapping={})
        assert '"mapping":{}' in m.canonical_json()

    def test_canonical_json_deterministic(self):
        m = SimpleModel(name="test", value=42)
        assert m.canonical_json() == m.canonical_json()

    def test_canonical_dict_excludes_none(self):
        m = SimpleModel(name="t", optional=None)
        d = m.canonical_dict()
        assert "optional" not in d

    def test_canonical_dict_includes_defaults(self):
        m = SimpleModel(name="t", value=0)
        d = m.canonical_dict()
        assert d["value"] == 0

    def test_canonical_json_unicode(self):
        m = SimpleModel(name="héllo", value=1)
        parsed = json.loads(m.canonical_json())
        assert parsed["name"] == "héllo"

    def test_canonical_json_special_chars(self):
        m = SimpleModel(name='line1\nline2\t"quote"', value=1)
        parsed = json.loads(m.canonical_json())
        assert parsed["name"] == 'line1\nline2\t"quote"'


class TestSHA256:
    def test_sha256_consistent(self):
        m = SimpleModel(name="test", value=42)
        assert m.sha256() == m.sha256()
        assert len(m.sha256()) == SHA256_HEX_LEN

    def test_sha256_different_for_different_values(self):
        a = SimpleModel(name="a", value=1)
        b = SimpleModel(name="b", value=2)
        assert a.sha256() != b.sha256()

    def test_sha256_hex_format(self):
        m = SimpleModel(name="t", value=1)
        h = m.sha256()
        assert isinstance(h, str)
        int(h, 16)

    def test_sha256_different_for_optional_field(self):
        a = SimpleModel(name="t", value=1)
        b = SimpleModel(name="t", value=1, optional="extra")
        assert a.sha256() != b.sha256()

    def test_sha256_nested_models(self):
        inner = SimpleModel(name="inner", value=99)
        a = NestedModel(inner=inner)
        b = NestedModel(inner=inner)
        assert a.sha256() == b.sha256()

    def test_sha256_repr_empty_list_vs_none(self):
        a = ModelWithOptionalList(names=None)
        b = ModelWithOptionalList(names=[])
        assert a.sha256() != b.sha256()


class TestRoundTrip:
    def test_round_trip_simple(self):
        m = SimpleModel(name="test", value=42)
        restored = SimpleModel.from_canonical_json(m.canonical_json())
        assert m == restored
        assert m.sha256() == restored.sha256()

    def test_round_trip_nested(self):
        inner = SimpleModel(name="inner", value=99)
        m = NestedModel(inner=inner, tag="mynested")
        restored = NestedModel.from_canonical_json(m.canonical_json())
        assert m == restored

    def test_round_trip_with_list(self):
        m = ModelWithList(items=[1, 2, 3], label="test")
        restored = ModelWithList.from_canonical_json(m.canonical_json())
        assert m == restored

    def test_round_trip_with_dict(self):
        m = ModelWithDict(mapping={"key": "value", "num": 42})
        restored = ModelWithDict.from_canonical_json(m.canonical_json())
        assert m == restored

    def test_round_trip_empty_dict(self):
        m = ModelWithDict(mapping={})
        restored = ModelWithDict.from_canonical_json(m.canonical_json())
        assert m == restored

    def test_round_trip_with_none_values(self):
        m = ModelWithOptionalList(names=None)
        restored = ModelWithOptionalList.from_canonical_json(m.canonical_json())
        assert m == restored

    def test_round_trip_nested_optional_present(self):
        m = ModelWithNestedOptional(data={"x": 1})
        restored = ModelWithNestedOptional.from_canonical_json(m.canonical_json())
        assert m == restored

    def test_round_trip_nested_optional_none(self):
        m = ModelWithNestedOptional()
        restored = ModelWithNestedOptional.from_canonical_json(m.canonical_json())
        assert m == restored

    def test_round_trip_preserves_types(self):
        m = ModelWithAllTypes(int_val=42, float_val=3.14, str_val="hello", bool_val=True)
        restored = ModelWithAllTypes.from_canonical_json(m.canonical_json())
        assert restored.int_val == 42
        assert restored.float_val == 3.14
        assert restored.str_val == "hello"
        assert restored.bool_val is True

    def test_from_canonical_json_with_extra_whitespace(self):
        m = SimpleModel(name="t", value=1)
        json_str = m.canonical_json()
        restored = SimpleModel.from_canonical_json(json_str)
        assert restored == m
