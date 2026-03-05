"""Tests for provider adapters."""

from unittest.mock import MagicMock

from driftlock.providers import AnthropicProvider, NormalizedUsage, OpenAIProvider


def _mock_openai_response(model="gpt-4o-mini", prompt=50, completion=20):
    r = MagicMock()
    r.model = model
    r.usage.prompt_tokens = prompt
    r.usage.completion_tokens = completion
    return r


def _mock_anthropic_response(model="claude-3-5-sonnet-20241022", input_tok=50, output_tok=20):
    r = MagicMock()
    r.model = model
    r.usage.input_tokens = input_tok
    r.usage.output_tokens = output_tok
    return r


def test_openai_provider_normalizes():
    p = OpenAIProvider()
    usage = p.normalize_response(_mock_openai_response(prompt=100, completion=40))
    assert isinstance(usage, NormalizedUsage)
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 40
    assert usage.model == "gpt-4o-mini"


def test_openai_provider_name():
    assert OpenAIProvider().provider_name() == "openai"


def test_anthropic_provider_normalizes():
    p = AnthropicProvider()
    usage = p.normalize_response(_mock_anthropic_response(input_tok=200, output_tok=80))
    assert isinstance(usage, NormalizedUsage)
    assert usage.prompt_tokens == 200
    assert usage.completion_tokens == 80
    assert usage.model == "claude-3-5-sonnet-20241022"


def test_anthropic_provider_name():
    assert AnthropicProvider().provider_name() == "anthropic"


def test_openai_provider_handles_missing_usage():
    p = OpenAIProvider()
    r = MagicMock()
    r.model = "gpt-4o"
    r.usage = None
    usage = p.normalize_response(r)
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0


def test_anthropic_provider_handles_missing_usage():
    p = AnthropicProvider()
    r = MagicMock()
    r.model = "claude-3-haiku-20240307"
    r.usage = None
    usage = p.normalize_response(r)
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
