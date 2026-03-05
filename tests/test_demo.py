"""
Tests for `driftlock demo` CLI subcommand.
No real API calls are made — all provider calls are mocked.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from driftlock.cli import main


@pytest.fixture(autouse=True)
def _stub_anthropic(monkeypatch):
    """
    Inject a minimal stub for the `anthropic` package if it isn't installed,
    so that `driftlock.anthropic_client` can be imported in any test.
    Forces a fresh import of anthropic_client so mocks take effect per test.
    """
    if "anthropic" not in sys.modules:
        stub = types.ModuleType("anthropic")
        stub.Anthropic = MagicMock
        stub.AsyncAnthropic = MagicMock
        monkeypatch.setitem(sys.modules, "anthropic", stub)
    monkeypatch.delitem(sys.modules, "driftlock.anthropic_client", raising=False)


# ------------------------------------------------------------------ #
# Shared mock factories
# ------------------------------------------------------------------ #

def _openai_response(model="gpt-4o-mini", prompt_tokens=15, completion_tokens=8):
    r = MagicMock()
    r.model = model
    r.usage.prompt_tokens = prompt_tokens
    r.usage.completion_tokens = completion_tokens
    r.usage.total_tokens = prompt_tokens + completion_tokens
    r.choices[0].message.content = "Hello there, nice to meet!"
    return r


def _anthropic_response(model="claude-haiku-4-5", input_tokens=15, output_tokens=8):
    r = MagicMock()
    r.model = model
    r.usage.input_tokens = input_tokens
    r.usage.output_tokens = output_tokens
    r.content = [MagicMock(text="Hello there, nice to meet!")]
    return r


# ------------------------------------------------------------------ #
# OpenAI path
# ------------------------------------------------------------------ #

class TestDemoOpenAI:
    def test_success(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        db = str(tmp_path / "driftlock.sqlite")

        with (
            patch("driftlock.client.OpenAI") as MockOpenAI,
            patch("driftlock.client.AsyncOpenAI"),
            patch("driftlock.cli._resolve_demo_db_path", return_value=db),
        ):
            mock_oi = MockOpenAI.return_value
            mock_oi.chat.completions.create.return_value = _openai_response()
            main(["demo"])

        out = capsys.readouterr().out
        assert "openai" in out
        assert "gpt-4o-mini" in out
        assert "Receipt" in out
        assert "driftlock stats" in out
        assert "driftlock recent" in out
        assert "driftlock forecast" in out

    def test_receipt_contains_db_path(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        db = str(tmp_path / "driftlock.sqlite")

        with (
            patch("driftlock.client.OpenAI") as MockOpenAI,
            patch("driftlock.client.AsyncOpenAI"),
            patch("driftlock.cli._resolve_demo_db_path", return_value=db),
        ):
            MockOpenAI.return_value.chat.completions.create.return_value = _openai_response()
            main(["demo"])

        assert db in capsys.readouterr().out

    def test_makes_one_request(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        db = str(tmp_path / "driftlock.sqlite")

        with (
            patch("driftlock.client.OpenAI") as MockOpenAI,
            patch("driftlock.client.AsyncOpenAI"),
            patch("driftlock.cli._resolve_demo_db_path", return_value=db),
        ):
            mock_oi = MockOpenAI.return_value
            mock_oi.chat.completions.create.return_value = _openai_response()
            main(["demo"])
            mock_oi.chat.completions.create.assert_called_once()

    def test_prompt_content(self, tmp_path, monkeypatch):
        """Verifies the demo sends the correct prompt."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        db = str(tmp_path / "driftlock.sqlite")

        with (
            patch("driftlock.client.OpenAI") as MockOpenAI,
            patch("driftlock.client.AsyncOpenAI"),
            patch("driftlock.cli._resolve_demo_db_path", return_value=db),
        ):
            mock_oi = MockOpenAI.return_value
            mock_oi.chat.completions.create.return_value = _openai_response()
            main(["demo"])
            call_kwargs = mock_oi.chat.completions.create.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs["messages"]
            assert any("5 words" in str(m.get("content", "")) for m in messages)


# ------------------------------------------------------------------ #
# Anthropic path
# ------------------------------------------------------------------ #

class TestDemoAnthropic:
    def test_success(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        db = str(tmp_path / "driftlock.sqlite")

        with (
            patch("driftlock.anthropic_client.Anthropic") as MockAnthropic,
            patch("driftlock.anthropic_client.AsyncAnthropic"),
            patch("driftlock.cli._resolve_demo_db_path", return_value=db),
        ):
            MockAnthropic.return_value.messages.create.return_value = _anthropic_response()
            main(["demo"])

        out = capsys.readouterr().out
        assert "anthropic" in out
        assert "claude-haiku-4-5" in out
        assert "Receipt" in out
        assert "driftlock stats" in out

    def test_makes_one_request(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        db = str(tmp_path / "driftlock.sqlite")

        with (
            patch("driftlock.anthropic_client.Anthropic") as MockAnthropic,
            patch("driftlock.anthropic_client.AsyncAnthropic"),
            patch("driftlock.cli._resolve_demo_db_path", return_value=db),
        ):
            mock_a = MockAnthropic.return_value
            mock_a.messages.create.return_value = _anthropic_response()
            main(["demo"])
            mock_a.messages.create.assert_called_once()


# ------------------------------------------------------------------ #
# Provider priority: OpenAI wins when both keys present
# ------------------------------------------------------------------ #

class TestDemoProviderPriority:
    def test_openai_wins_when_both_keys_set(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        db = str(tmp_path / "driftlock.sqlite")

        with (
            patch("driftlock.client.OpenAI") as MockOpenAI,
            patch("driftlock.client.AsyncOpenAI"),
            patch("driftlock.cli._resolve_demo_db_path", return_value=db),
        ):
            MockOpenAI.return_value.chat.completions.create.return_value = _openai_response()
            main(["demo"])

        out = capsys.readouterr().out
        assert "openai" in out


# ------------------------------------------------------------------ #
# No API key
# ------------------------------------------------------------------ #

class TestDemoNoKey:
    def test_exits_with_error_when_no_key(self, monkeypatch, capsys):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(SystemExit) as exc_info:
            main(["demo"])

        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "API key" in err

    def test_error_message_is_actionable(self, monkeypatch, capsys):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(SystemExit):
            main(["demo"])

        err = capsys.readouterr().err
        assert "OPENAI_API_KEY" in err
        assert "ANTHROPIC_API_KEY" in err


# ------------------------------------------------------------------ #
# Policy block handling
# ------------------------------------------------------------------ #

class TestDemoPolicyBlock:
    def test_policy_block_does_not_crash(self, tmp_path, capsys, monkeypatch):
        """If a policy blocks the demo, print reason + guidance and exit 1."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        db = str(tmp_path / "driftlock.sqlite")

        from driftlock.policy import PolicyViolationError, RuleDecision

        with (
            patch("driftlock.client.OpenAI") as MockOpenAI,
            patch("driftlock.client.AsyncOpenAI"),
            patch("driftlock.cli._resolve_demo_db_path", return_value=db),
        ):
            MockOpenAI.return_value.chat.completions.create.side_effect = PolicyViolationError(
                "MonthlyBudgetRule",
                RuleDecision(
                    allow=False,
                    action="block",
                    metadata={"spent_usd": 50.0, "max_usd": 50.0},
                ),
            )
            with pytest.raises(SystemExit) as exc_info:
                main(["demo"])

        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "MonthlyBudgetRule" in out
        assert "MonthlyBudgetRule" in out or "loosen" in out.lower()

    def test_policy_block_shows_how_to_fix(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        db = str(tmp_path / "driftlock.sqlite")

        from driftlock.policy import PolicyViolationError, RuleDecision

        with (
            patch("driftlock.client.OpenAI") as MockOpenAI,
            patch("driftlock.client.AsyncOpenAI"),
            patch("driftlock.cli._resolve_demo_db_path", return_value=db),
        ):
            MockOpenAI.return_value.chat.completions.create.side_effect = PolicyViolationError(
                "VelocityLimitRule",
                RuleDecision(allow=False, action="block", metadata={}),
            )
            with pytest.raises(SystemExit):
                main(["demo"])

        out = capsys.readouterr().out
        assert "VelocityLimitRule" in out


# ------------------------------------------------------------------ #
# DB path resolution
# ------------------------------------------------------------------ #

class TestResolveDemoDbPath:
    def test_uses_cwd_when_writable(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from driftlock.cli import _resolve_demo_db_path
        result = _resolve_demo_db_path()
        assert result == str(tmp_path / "driftlock.sqlite") or result == "driftlock.sqlite"

    def test_falls_back_to_home_when_cwd_not_writable(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        def _raise(*a, **kw):
            raise OSError("read only")

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: fake_home))

        with patch("pathlib.Path.touch", side_effect=_raise):
            from driftlock.cli import _resolve_demo_db_path
            result = _resolve_demo_db_path()

        assert "driftlock.sqlite" in result
        assert str(fake_home) in result
