import time
from collections.abc import Callable
from typing import Any


RETRY_DELAYS = (2, 4)


class GroqTransientError(RuntimeError):
    """A Groq request failed after the bounded transient-error retries."""


def call_with_retry(operation: Callable[[], Any]) -> Any:
    try:
        import groq
        import httpx
    except ImportError:
        raise

    transient_errors = tuple(
        error_type
        for error_type in (
            getattr(groq, "APIConnectionError", None),
            getattr(groq, "APITimeoutError", None),
            getattr(httpx, "NetworkError", None),
            getattr(httpx, "TimeoutException", None),
        )
        if isinstance(error_type, type)
    )

    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            return operation()
        except transient_errors as exc:
            if attempt >= len(RETRY_DELAYS):
                raise GroqTransientError(
                    f"Groq request failed after {attempt + 1} attempts: {exc}"
                ) from exc
            time.sleep(RETRY_DELAYS[attempt])

    raise AssertionError("unreachable")
