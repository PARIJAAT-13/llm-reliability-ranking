"""Tests for CITATION.cff generation."""

from pathlib import Path

from llm_reliability.reproducibility.citation import CitationGenerator


class TestCitationGenerator:
    def test_build_defaults(self) -> None:
        gen = CitationGenerator()
        cff = gen.build(experiment_name="Test Experiment")
        assert cff["cff-version"] == "1.2.0"
        assert cff["title"] == "Test Experiment"
        assert cff["version"] == "0.1.0"
        assert cff["type"] == "software"
        assert len(cff["authors"]) == 1
        assert "repository-code" in cff
        assert "license" in cff
        assert "keywords" in cff

    def test_build_custom_params(self) -> None:
        gen = CitationGenerator()
        authors = [{"family-names": "Smith", "given-names": "John"}]
        cff = gen.build(
            experiment_name="My Software",
            version="2.0.0",
            abstract="A test software.",
            authors=authors,
            repository_url="https://github.com/test/repo",
            license_name="Apache-2.0",
        )
        assert cff["title"] == "My Software"
        assert cff["version"] == "2.0.0"
        assert cff["abstract"] == "A test software."
        assert cff["authors"] == authors
        assert cff["repository-code"] == "https://github.com/test/repo"
        assert cff["license"] == "Apache-2.0"

    def test_build_has_date_released(self) -> None:
        gen = CitationGenerator()
        cff = gen.build(experiment_name="Test")
        assert "date-released" in cff
        assert len(cff["date-released"]) == 10

    def test_to_yaml_contains_required_fields(self) -> None:
        gen = CitationGenerator()
        cff = gen.build(experiment_name="Test")
        yaml_str = gen.to_yaml(cff)
        assert "cff-version:" in yaml_str
        assert "title: Test" in yaml_str
        assert "version: '0.1.0'" in yaml_str or "version: 0.1.0" in yaml_str
        assert "authors:" in yaml_str

    def test_save_creates_file(self, tmp_path: Path) -> None:
        gen = CitationGenerator()
        cff = gen.build(experiment_name="Test")
        dest = gen.save(cff, str(tmp_path / "CITATION.cff"))
        assert dest.exists()
        content = dest.read_text(encoding="utf-8")
        assert "title: Test" in content

    def test_multiple_authors(self) -> None:
        gen = CitationGenerator()
        authors = [
            {"family-names": "Smith", "given-names": "John"},
            {"family-names": "Doe", "given-names": "Jane"},
        ]
        cff = gen.build(experiment_name="Test", authors=authors)
        assert len(cff["authors"]) == 2

    def test_keywords_include_llm_reliability(self) -> None:
        gen = CitationGenerator()
        cff = gen.build(experiment_name="Test")
        assert "LLM" in cff["keywords"]
        assert "reliability" in cff["keywords"]
        assert "ranking" in cff["keywords"]
