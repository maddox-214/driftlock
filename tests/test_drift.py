"""Tests for prompt drift tracking."""

from driftlock.drift import detect_drift, hash_prompt


def test_hash_prompt_returns_string():
    h = hash_prompt([{"role": "user", "content": "Tell me about {topic}"}])
    assert isinstance(h, str)
    assert len(h) == 16  # 16-char hex prefix


def test_hash_prompt_same_input_same_hash():
    msgs = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello"}]
    assert hash_prompt(msgs) == hash_prompt(msgs)


def test_hash_prompt_different_system_different_hash():
    msgs_a = [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "Hello"}]
    msgs_b = [{"role": "system", "content": "Be verbose."}, {"role": "user", "content": "Hello"}]
    assert hash_prompt(msgs_a) != hash_prompt(msgs_b)


def test_hash_prompt_only_first_user_message():
    """Two calls with same system + first user but different second user should have same hash."""
    msgs_a = [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Template {x}"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "dynamic input A"},
    ]
    msgs_b = [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Template {x}"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "dynamic input B"},
    ]
    assert hash_prompt(msgs_a) == hash_prompt(msgs_b)


def test_hash_prompt_with_system_kwarg():
    h1 = hash_prompt([], system="You are helpful.")
    h2 = hash_prompt([], system="You are helpful.")
    assert h1 == h2

    h3 = hash_prompt([], system="You are strict.")
    assert h1 != h3


def test_hash_prompt_returns_none_for_empty():
    assert hash_prompt([]) is None
    assert hash_prompt([], system=None) is None


def test_detect_drift_no_changes():
    # Same hash throughout — no changes
    history = [
        {"timestamp": "2025-01-03T00:00:00", "prompt_hash": "abc123", "prompt_tokens": 100},
        {"timestamp": "2025-01-02T00:00:00", "prompt_hash": "abc123", "prompt_tokens": 100},
        {"timestamp": "2025-01-01T00:00:00", "prompt_hash": "abc123", "prompt_tokens": 100},
    ]
    changes = detect_drift(history)
    assert len(changes) == 1  # first appearance counts as a "change" from None


def test_detect_drift_finds_changes():
    history = [
        {"timestamp": "2025-01-03T00:00:00", "prompt_hash": "def456", "prompt_tokens": 150},
        {"timestamp": "2025-01-02T00:00:00", "prompt_hash": "abc123", "prompt_tokens": 100},
        {"timestamp": "2025-01-01T00:00:00", "prompt_hash": "abc123", "prompt_tokens": 100},
    ]
    changes = detect_drift(history)
    # First appearance + one actual change
    assert len(changes) == 2
    last = changes[-1]
    assert last["old_hash"] == "abc123"
    assert last["new_hash"] == "def456"


def test_drift_recorded_in_client(tmp_path):
    from unittest.mock import patch

    from driftlock import DriftlockClient, DriftlockConfig
    from unittest.mock import MagicMock

    def _mock_response():
        r = MagicMock()
        r.model = "gpt-4o-mini"
        r.usage.prompt_tokens = 20
        r.usage.completion_tokens = 5
        r.usage.total_tokens = 25
        return r

    config = DriftlockConfig(db_path=str(tmp_path / "test.db"), log_json=False)
    with patch("driftlock.client.OpenAI") as MockOpenAI, patch("driftlock.client.AsyncOpenAI"):
        mock = MockOpenAI.return_value
        mock.chat.completions.create.return_value = _mock_response()
        c = DriftlockClient(api_key="sk-test", config=config)

        c.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Summarize: {text}"},
            ],
            _dl_endpoint="summarize",
        )

    recent = c.recent_calls(limit=1)
    assert recent[0]["prompt_hash"] is not None
