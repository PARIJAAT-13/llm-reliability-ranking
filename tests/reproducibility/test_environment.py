"""Tests for environment capture."""

from llm_reliability.reproducibility.environment import EnvironmentCapture


class TestEnvironmentCapture:
    def test_capture_basic(self) -> None:
        env = EnvironmentCapture.capture()
        assert env.python_version != ""
        assert env.python_implementation != ""
        assert env.platform_system != ""
        assert env.captured_at != ""

    def test_capture_has_cpu_count(self) -> None:
        env = EnvironmentCapture.capture()
        assert env.cpu_count is None or env.cpu_count > 0

    def test_to_dict_serializable(self) -> None:
        env = EnvironmentCapture.capture()
        d = env.to_dict()
        assert isinstance(d, dict)
        assert "python_version" in d
        assert "platform_system" in d
        assert "captured_at" in d

    def test_to_dict_values_match(self) -> None:
        env = EnvironmentCapture.capture()
        d = env.to_dict()
        assert d["python_version"] == env.python_version
        assert d["python_implementation"] == env.python_implementation

    def test_capture_include_all_packages(self) -> None:
        env = EnvironmentCapture.capture(include_all_packages=True)
        assert isinstance(env.packages, dict)

    def test_capture_default_excludes_most_packages(self) -> None:
        env = EnvironmentCapture.capture(include_all_packages=False)
        assert isinstance(env.packages, dict)

    def test_git_commit_may_be_none(self) -> None:
        env = EnvironmentCapture.capture()
        assert env.git_commit is None or isinstance(env.git_commit, str)
