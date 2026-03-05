"""Tests for the alert system."""

from unittest.mock import MagicMock, patch

import pytest

from driftlock.alerts import (
    ALERT_COST_WARNING,
    ALERT_POLICY_BLOCK,
    LogAlertChannel,
    SlackAlertChannel,
    WebhookAlertChannel,
    fire_alert,
)


def test_log_alert_channel_sends():
    ch = LogAlertChannel()
    # Should not raise
    ch.send(ALERT_POLICY_BLOCK, {"rule": "MaxCostPerRequestRule", "cost": 0.5})


def test_fire_alert_calls_all_channels():
    ch1 = MagicMock()
    ch2 = MagicMock()
    fire_alert([ch1, ch2], ALERT_COST_WARNING, {"model": "gpt-4o", "cost_usd": 0.1})
    ch1.send.assert_called_once_with(ALERT_COST_WARNING, {"model": "gpt-4o", "cost_usd": 0.1})
    ch2.send.assert_called_once_with(ALERT_COST_WARNING, {"model": "gpt-4o", "cost_usd": 0.1})


def test_fire_alert_swallows_channel_exceptions():
    ch = MagicMock()
    ch.send.side_effect = RuntimeError("network down")
    # Should not propagate
    fire_alert([ch], ALERT_POLICY_BLOCK, {"rule": "test"})


def test_webhook_channel_posts(tmp_path):
    ch = WebhookAlertChannel(url="https://example.com/hook")
    with patch("urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        ch.send(ALERT_POLICY_BLOCK, {"rule": "VelocityLimitRule"})
    mock_open.assert_called_once()


def test_slack_channel_posts():
    ch = SlackAlertChannel(webhook_url="https://hooks.slack.com/test")
    with patch("urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        ch.send(ALERT_COST_WARNING, {"cost_usd": 1.23})
    mock_open.assert_called_once()


def test_webhook_channel_swallows_network_error():
    ch = WebhookAlertChannel(url="https://example.com/hook")
    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        # Should not raise
        ch.send(ALERT_POLICY_BLOCK, {"rule": "test"})


def test_alert_fires_on_policy_block(tmp_path):
    """Alert channel receives event when a policy blocks a request."""
    from unittest.mock import patch

    from driftlock import DriftlockClient, DriftlockConfig
    from driftlock.policy import PolicyEngine, RestrictModelRule

    alert_ch = MagicMock()
    config = DriftlockConfig(
        db_path=str(tmp_path / "test.db"),
        log_json=False,
        alert_channels=[alert_ch],
    )
    policy = PolicyEngine([RestrictModelRule(disallowed_models={"gpt-4o"})])

    with patch("driftlock.client.OpenAI"), patch("driftlock.client.AsyncOpenAI"):
        c = DriftlockClient(api_key="sk-test", config=config, policy=policy)
        with pytest.raises(Exception):
            c.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
            )

    alert_ch.send.assert_called_once()
    event_type = alert_ch.send.call_args[0][0]
    assert event_type == ALERT_POLICY_BLOCK
