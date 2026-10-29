"""The reasoning loop alone decides retries, from what the client reports."""

import httpx
import pytest

from max_ai.base.reasoning import _MAX_RETRY_DELAY, _retry_delay
from max_ai.capabilities.clients.openai import OpenAIChatCompletionClient
from max_ai.errors.client import ClientError


@pytest.mark.parametrize("error", [
    ClientError.rate_limit_exceeded(),
    ClientError.request_timeout(5),
    ClientError.stream_interrupted(),
    ClientError.api_error("openai"),               # no status: connection error
    ClientError.api_error("openai", status=408),
    ClientError.api_error("openai", status=409),
    ClientError.api_error("openai", status=500),
    ClientError.api_error("openai", status=503),
    ClientError.provider_error("x", status_code=502),
])
def test_transient_errors_are_retried(error):
    assert _retry_delay(error, 0) is not None


@pytest.mark.parametrize("error", [
    ClientError.api_error("openai", status=403),
    ClientError.api_error("openai", status=404),
    ClientError.api_error("openai", status=422),
    ClientError.authentication_failed(),
    ClientError.invalid_request("bad"),
    ValueError("not a client error"),
])
def test_permanent_errors_fail_at_once(error):
    assert _retry_delay(error, 0) is None


def test_the_providers_retry_after_wins_but_is_capped():
    assert _retry_delay(ClientError.rate_limit_exceeded(retry_after=20), 0) == 20
    assert _retry_delay(ClientError.rate_limit_exceeded(retry_after=999), 0) == _MAX_RETRY_DELAY


def test_backoff_grows_with_jitter():
    error = ClientError.request_timeout(5)
    delays = {_retry_delay(error, 2) for _ in range(20)}
    assert all(2.0 <= d <= 6.0 for d in delays)  # 4s * [0.5, 1.5]
    assert len(delays) > 1


def test_the_openai_sdk_does_not_retry_on_its_own():
    client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key="sk-test")
    assert client.client.max_retries == 0


def test_openai_errors_carry_status_and_retry_after():
    from openai import NotFoundError, RateLimitError

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    limited = RateLimitError("slow down", body=None, response=httpx.Response(
        429, headers={"retry-after": "7"}, request=request,
    ))
    error = OpenAIChatCompletionClient._map_openai_error(limited)
    assert (error.kind, error.status, error.retry_after) == ("rate_limit", 429, 7.0)

    missing = NotFoundError("no such model", body=None, response=httpx.Response(
        404, request=request,
    ))
    error = OpenAIChatCompletionClient._map_openai_error(missing)
    assert (error.kind, error.status) == ("api_error", 404)
    assert _retry_delay(error, 0) is None
