"""
Purpose
-------
Provide a production-ready GPT agent that integrates with the LLM Reliability
Ranking framework's existing pipeline without modifying any framework component.

Responsibilities
----------------
- Implement every method required by the ``Agent`` abstract interface
- Accept ``Configuration`` as the sole constructor argument
- Read the OpenAI API key exclusively from the ``OPENAI_API_KEY`` environment
  variable — no credentials are ever hardcoded or stored in configuration files
- Extract a prompt from the task dict, construct an ``LLMRequest``, and use the
  ``BaseLLMAdapter`` infrastructure (validation, latency measurement, retry,
  rate limiting) to call the OpenAI Chat Completions API
- Return ONLY the raw model output string from ``run()``; the benchmark and
  ``ExperimentRunner`` are responsible for wrapping it in ``ExecutionRecord``
- Register with ``ProviderRegistry`` under the name ``"openai"``

Usage example
-------------
>>> import os
>>> os.environ["OPENAI_API_KEY"] = "sk-..."   # set before instantiating
>>> from llm_reliability.configs.config import Configuration
>>> from llm_reliability.agents.gpt_agent import GPTAgent
>>> cfg = Configuration(
...     experiment_name="pilot",
...     benchmark="AgentBoard",
...     agent="GPTAgent",
...     llm="gpt-4.1",
...     prompt_version="v1",
...     dataset_version="1.0",
...     seed=42,
...     repetitions=3,
... )
>>> agent = GPTAgent(cfg)
>>> agent.initialize()          # authenticates and warms up the client
>>> task = {
...     "task_id": "t1",
...     "prompt": "What is 2 + 2?",
... }
>>> answer = agent.run(task)    # returns raw model output string
>>> agent.shutdown()

Design notes
------------
Two-layer architecture
~~~~~~~~~~~~~~~~~~~~~~
The ``Agent`` interface and the ``BaseLLMAdapter`` interface are parallel
hierarchies in this framework:

* ``Agent``         — what the ``ExperimentRunner`` and benchmarks see
* ``BaseLLMAdapter`` — what the provider-level infrastructure (retry, logging,
                       validation, rate limiting) provides

``GPTAgent`` bridges both:

  GPTAgent(Agent)
    └── _adapter: _OpenAIAdapter(BaseLLMAdapter)

``GPTAgent.run()`` delegates all OpenAI HTTP work to ``_OpenAIAdapter``,
inheriting request validation, latency measurement, retry-with-back-off, and
structured logging for free.

Deferred import of ``openai``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``openai`` package is imported inside ``_OpenAIAdapter.initialize()`` rather
than at module scope.  This means:

1. Tests that mock the HTTP layer can import this module without having a live
   API key or a running OpenAI server.
2. The module loads instantly even when the ``openai`` package is not installed;
   the ``ImportError`` surfaces only when ``initialize()`` is called.

Configuration-driven model selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The OpenAI model identifier is taken from ``config.llm`` with a fall-back to
``"gpt-4.1"``.  Temperature, max_tokens, and seed are read from
``config.metadata`` so they can be set per-experiment without touching the
``Configuration`` schema.

Prompt extraction
~~~~~~~~~~~~~~~~~
``run(task)`` looks for the prompt in this priority order:

1. ``task["prompt"]``          — standard AgentBoard / GAIA / SWEBench key
2. ``task["question"]``        — GAIA uses this field name
3. ``task["problem_statement"]``— SWE-bench may use this key
4. ``str(task)``               — last resort: stringify the whole dict

This makes ``GPTAgent`` compatible with every existing benchmark adapter
without requiring changes to those adapters.

Environment variables
~~~~~~~~~~~~~~~~~~~~~
``OPENAI_API_KEY``    (required)  — OpenAI secret key; validated in initialize()
``OPENAI_BASE_URL``   (optional)  — override base URL for Azure OpenAI or proxies
``OPENAI_ORG_ID``     (optional)  — organization ID for billing attribution
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from llm_reliability.agents.adapters.base_llm_adapter import BaseLLMAdapter
from llm_reliability.agents.adapters.exceptions import (
    AuthenticationError,
)
from llm_reliability.agents.adapters.exceptions import (
    ConnectionError as ProviderConnectionError,
)
from llm_reliability.agents.adapters.exceptions import (
    ProviderError,
    RateLimitError,
    ResponseValidationError,
)
from llm_reliability.agents.adapters.provider_registry import ProviderRegistry
from llm_reliability.agents.adapters.request_models import LLMRequest
from llm_reliability.agents.adapters.response_models import LLMResponse
from llm_reliability.agents.utils.rate_limiter import RateLimiter
from llm_reliability.configs.config import Configuration
from llm_reliability.runtime import Runtime
from llm_reliability.runtime.registry import RuntimeRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = "gpt-4.1"
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_MAX_TOKENS: int = 1024
DEFAULT_TOP_P: float = 1.0
DEFAULT_REQUESTS_PER_SECOND: float = 3.0  # conservative default; raise in config
GPT_AGENT_VERSION: str = "1.0"

# Keys tried in order when extracting the prompt from a task dict
_PROMPT_KEYS: tuple[str, ...] = ("prompt", "question", "problem_statement")


# ---------------------------------------------------------------------------
# Private: OpenAI provider adapter (BaseLLMAdapter subclass)
# ---------------------------------------------------------------------------


class _OpenAIAdapter(BaseLLMAdapter):
    """Internal OpenAI Chat Completions adapter.

    Wraps the ``openai`` SDK and converts exceptions into the framework's
    typed ``ProviderError`` hierarchy so ``BaseLLMAdapter.retry()`` can
    handle them uniformly.

    This class is intentionally private (``_`` prefix).  External callers
    should use ``GPTAgent``, never this class directly.
    """

    # The openai client is stored as an instance attribute set in initialize().
    # Type annotation uses string to avoid importing openai at module scope.
    _client: Any

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)
        self._client = None
        self._model = config.metadata.get("model", config.llm) or DEFAULT_MODEL
        self._temperature = float(config.metadata.get("temperature", DEFAULT_TEMPERATURE))
        self._max_tokens = int(config.metadata.get("max_tokens", DEFAULT_MAX_TOKENS))
        self._top_p = float(config.metadata.get("top_p", DEFAULT_TOP_P))
        self._system_prompt: str | None = config.metadata.get("system_prompt")

    # ------------------------------------------------------------------
    # BaseLLMAdapter abstract methods
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Authenticate with OpenAI and create the SDK client.

        Reads ``OPENAI_API_KEY`` from the environment.  Raises
        ``AuthenticationError`` if the variable is not set.

        The ``openai`` package is imported here (not at module scope) so that
        the module can be imported in test environments that do not have a live
        OpenAI connection.
        """
        try:
            import openai  # noqa: PLC0415 — intentional deferred import
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required to use GPTAgent. "
                "Install it with: pip install openai>=1.0"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it before calling initialize()."
            )

        base_url: str | None = os.environ.get("OPENAI_BASE_URL")
        org_id: str | None = os.environ.get("OPENAI_ORG_ID")

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if org_id:
            kwargs["organization"] = org_id

        self._client = openai.OpenAI(**kwargs)
        logger.info(
            "OpenAI client initialised (model=%s, temperature=%.2f, max_tokens=%d).",
            self._model,
            self._temperature,
            self._max_tokens,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Call the OpenAI Chat Completions API and return an ``LLMResponse``.

        Maps all OpenAI-specific exceptions to the framework's typed
        ``ProviderError`` hierarchy so the retry infrastructure in
        ``BaseLLMAdapter`` can handle them correctly.

        Parameters
        ----------
        request:
            Validated ``LLMRequest`` built from the task prompt and config.

        Returns
        -------
        LLMResponse
            Structured, provider-agnostic response ready for the framework.

        Raises
        ------
        AuthenticationError
            On HTTP 401 / invalid API key.
        RateLimitError
            On HTTP 429 / token-limit exceeded.
        ProviderConnectionError
            On network-level failures.
        ResponseValidationError
            If the API returns an empty or malformed completion.
        ProviderError
            For any other OpenAI API error.
        """
        if self._client is None:
            raise RuntimeError(
                "_OpenAIAdapter.generate() called before initialize(). Call initialize() first."
            )

        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("openai package not available.") from exc

        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        elif self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
        }
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.stop_sequences:
            kwargs["stop"] = request.stop_sequences

        t0 = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as exc:
            logger.error("OpenAI authentication failed: %s", exc)
            raise AuthenticationError(f"Authentication failed: {exc}") from exc
        except openai.RateLimitError as exc:
            logger.warning("OpenAI rate limit hit: %s", exc)
            raise RateLimitError(f"Rate limit exceeded: {exc}") from exc
        except openai.APIConnectionError as exc:
            logger.error("OpenAI network error: %s", exc)
            raise ProviderConnectionError(f"Network error: {exc}") from exc
        except openai.APITimeoutError as exc:
            logger.error("OpenAI request timed out: %s", exc)
            raise ProviderConnectionError(f"Request timed out: {exc}") from exc
        except openai.BadRequestError as exc:
            logger.error("OpenAI bad request: %s", exc)
            raise ProviderError(f"Bad request: {exc}") from exc
        except openai.APIError as exc:
            logger.error("OpenAI API error: %s", exc)
            raise ProviderError(f"API error: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0

        choice = completion.choices[0] if completion.choices else None
        if choice is None or not getattr(choice.message, "content", None):
            raise ResponseValidationError(
                f"OpenAI returned an empty or missing completion choice. Response: {completion}"
            )

        text: str = choice.message.content or ""
        finish_reason: str = choice.finish_reason or "unknown"
        usage = completion.usage

        return LLMResponse(
            text=text,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            tokens_input=usage.prompt_tokens if usage else 0,
            tokens_output=usage.completion_tokens if usage else 0,
            model_name=completion.model or self._model,
            provider="openai",
            metadata={
                "id": completion.id,
                "system_fingerprint": getattr(completion, "system_fingerprint", None),
            },
        )

    def shutdown(self) -> None:
        """Release the OpenAI client connection."""
        if self._client is not None:
            # The openai v1 SDK uses httpx under the hood; closing releases
            # the underlying connection pool.
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
            logger.debug("OpenAI client shut down.")

    def provider_metadata(self) -> dict[str, Any]:
        """Return OpenAI provider metadata."""
        return {
            "provider": "openai",
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "top_p": self._top_p,
        }

    def health_check(self) -> bool:
        """Return True if the OpenAI API is reachable.

        Issues a minimal models-list request.  Returns False on any error
        rather than propagating — callers can decide how to handle a
        degraded provider.
        """
        if self._client is None:
            return False
        try:
            self._client.models.list()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAI health check failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Public: GPTAgent — Agent interface implementation
# ---------------------------------------------------------------------------


class GPTAgent(Runtime):
    """GPT agent for the LLM Reliability Ranking framework.

    Integrates with the existing ``ExperimentRunner`` / ``Benchmark`` pipeline
    without requiring any changes to framework components.

    Parameters
    ----------
    config:
        Framework ``Configuration`` object.  The following ``metadata`` keys
        are recognised (all optional):

        ``model``              (str)   — OpenAI model identifier.
                                         Default: ``"gpt-4.1"``.
        ``temperature``        (float) — Sampling temperature in [0, 2].
                                         Default: ``0.0``.
        ``max_tokens``         (int)   — Max tokens to generate.
                                         Default: ``1024``.
        ``top_p``              (float) — Nucleus sampling probability.
                                         Default: ``1.0``.
        ``system_prompt``      (str)   — Optional system-level instructions.
        ``max_retries``        (int)   — Max retry attempts on transient errors.
                                         Default: ``3``.
        ``retry_backoff``      (float) — Base back-off seconds for retries.
                                         Default: ``1.0``.
        ``requests_per_second``(float) — Rate-limit cap.
                                         Default: ``3.0``.

    Environment variables
    ---------------------
    ``OPENAI_API_KEY``    Required — OpenAI secret key.
    ``OPENAI_BASE_URL``   Optional — Override API base URL (Azure, proxies).
    ``OPENAI_ORG_ID``     Optional — Organization ID.

    Raises
    ------
    ValueError
        If ``config`` is ``None``.
    """

    def __init__(self, config: Configuration) -> None:
        """Initialise GPTAgent with framework configuration.

        Does NOT import ``openai`` or touch the network — deferred to
        ``initialize()``.
        """
        if config is None:
            raise ValueError("Configuration must be provided to GPTAgent.")

        self._config = config
        self._adapter = _OpenAIAdapter(config)

        self._max_retries: int = int(config.metadata.get("max_retries", 3))
        self._retry_backoff: float = float(config.metadata.get("retry_backoff", 1.0))
        self._rate_limiter = RateLimiter(
            requests_per_second=float(
                config.metadata.get("requests_per_second", DEFAULT_REQUESTS_PER_SECOND)
            )
        )

        logger.debug(
            "GPTAgent created (model=%s, seed=%d, retries=%d).",
            config.metadata.get("model", config.llm) or DEFAULT_MODEL,
            config.seed,
            self._max_retries,
        )

    # ------------------------------------------------------------------
    # Agent interface — mandatory abstract methods
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Authenticate with OpenAI and warm up the SDK client.

        Reads ``OPENAI_API_KEY`` from the environment.

        Raises
        ------
        AuthenticationError
            If ``OPENAI_API_KEY`` is not set.
        ImportError
            If the ``openai`` package is not installed.
        """
        logger.info("Initialising GPTAgent.")
        self._adapter.initialize()
        logger.info(
            "GPTAgent ready (model=%s).",
            self._adapter._model,
        )

    def reset(self) -> None:
        """Reset per-task state.

        The OpenAI API is stateless between calls; this clears the internal
        request/response logs accumulated by ``BaseLLMAdapter`` so that each
        task starts with a clean audit trail.
        """
        self._adapter._request_logs.clear()
        self._adapter._response_logs.clear()
        logger.debug("GPTAgent state reset.")

    def run(self, task: dict[str, Any]) -> Any:
        """Execute the GPT model on a benchmark task.

        Extracts the prompt from the task dict, constructs an ``LLMRequest``
        with configuration-driven parameters, calls the OpenAI API through the
        adapter's retry and rate-limiting infrastructure, and returns the raw
        model output string.

        Does NOT create an ``ExecutionRecord`` — that is the responsibility of
        the benchmark's ``run()`` method and the ``ExperimentRunner``.

        Parameters
        ----------
        task:
            Task payload dict as returned by a benchmark adapter's
            ``get_task()``.  Must contain at least one of: ``prompt``,
            ``question``, ``problem_statement``.

        Returns
        -------
        str
            Raw text output from the GPT model.

        Raises
        ------
        ValueError
            If no prompt key can be found in the task dict.
        AuthenticationError, RateLimitError, ProviderConnectionError, ProviderError
            Propagated from the OpenAI adapter after all retries are exhausted.
        """
        prompt = self._extract_prompt(task)

        request = LLMRequest(
            prompt=prompt,
            temperature=float(self._config.metadata.get("temperature", DEFAULT_TEMPERATURE)),
            max_tokens=int(self._config.metadata.get("max_tokens", DEFAULT_MAX_TOKENS)),
            top_p=float(self._config.metadata.get("top_p", DEFAULT_TOP_P)),
            seed=self._config.seed if self._config.seed is not None else None,
            system_prompt=self._config.metadata.get("system_prompt"),
        )

        logger.info(
            "GPTAgent.run: task_id=%r, model=%s, prompt_len=%d.",
            task.get("task_id", "<unknown>"),
            self._adapter._model,
            len(prompt),
        )

        # Enforce rate limit before issuing the API call
        self._rate_limiter.acquire()

        t0 = time.perf_counter()
        response = self._adapter.retry(
            request,
            max_attempts=self._max_retries,
            backoff_seconds=self._retry_backoff,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        logger.info(
            "GPTAgent.run complete: task_id=%r, finish=%s, latency=%.1fms, "
            "tokens_in=%d, tokens_out=%d.",
            task.get("task_id", "<unknown>"),
            response.finish_reason,
            latency_ms,
            response.tokens_input,
            response.tokens_output,
        )

        return response.text

    def shutdown(self) -> None:
        """Release the OpenAI client and connection pool."""
        logger.info("Shutting down GPTAgent.")
        self._adapter.shutdown()

    def metadata(self) -> dict[str, Any]:
        """Return descriptive metadata for logging and reproducibility.

        Returns
        -------
        dict[str, Any]
            Includes agent name, provider, model, version, and key parameters.
        """
        return {
            "name": "GPTAgent",
            "provider": "openai",
            "model": self._adapter._model,
            "version": GPT_AGENT_VERSION,
            "temperature": self._adapter._temperature,
            "max_tokens": self._adapter._max_tokens,
            "top_p": self._adapter._top_p,
            "seed": self._config.seed,
            "max_retries": self._max_retries,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_prompt(task: dict[str, Any]) -> str:
        """Extract the user prompt from a benchmark task dict.

        Tries keys in priority order:
        ``prompt`` → ``question`` → ``problem_statement`` → ``str(task)``

        Parameters
        ----------
        task:
            Task payload dict from a benchmark adapter.

        Returns
        -------
        str
            Non-empty prompt string.

        Raises
        ------
        ValueError
            If the task dict is empty and has no usable prompt.
        """
        for key in _PROMPT_KEYS:
            value = task.get(key)
            if value and str(value).strip():
                return str(value).strip()

        # Last resort: use the entire task dict serialised as a string
        fallback = str(task).strip()
        if not fallback or fallback == "{}":
            raise ValueError(
                f"Cannot extract a prompt from task dict: {task!r}. "
                f"Task must contain one of: {_PROMPT_KEYS}."
            )

        logger.warning(
            "No standard prompt key found in task %r; using str(task) as prompt.",
            task.get("task_id", "<unknown>"),
        )
        return fallback


# ---------------------------------------------------------------------------
# Registry self-registration
#
# Register under "openai" in ProviderRegistry so that the pipeline can
# discover the underlying adapter by provider name.
#
# Guard against double-registration if the module is re-imported.
# ---------------------------------------------------------------------------
if not ProviderRegistry.exists("openai"):
    ProviderRegistry.register("openai", _OpenAIAdapter)
if not RuntimeRegistry.exists("gpt"):
    RuntimeRegistry.register("gpt", GPTAgent)
