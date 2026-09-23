from unittest.mock import Mock, patch

import httpx
import pytest

from src.study_assistant.groq_retry import GroqTransientError, call_with_retry


def test_retries_connection_timeout_with_short_backoff():
    response = object()
    operation = Mock(side_effect=[httpx.ConnectTimeout("connect timeout"), response])

    with patch("src.study_assistant.groq_retry.time.sleep") as sleep:
        assert call_with_retry(operation) is response

    assert operation.call_count == 2
    sleep.assert_called_once_with(2)


def test_stops_after_two_retries_and_preserves_error_context():
    operation = Mock(side_effect=httpx.ConnectTimeout("connect timeout"))

    with patch("src.study_assistant.groq_retry.time.sleep") as sleep:
        with pytest.raises(GroqTransientError, match="after 3 attempts"):
            call_with_retry(operation)

    assert operation.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [2, 4]
