"""
Alert channels for Driftlock governance events.

Alerts fire when:
  - A policy rule blocks a request (PolicyViolationError)
  - A cost warning threshold is crossed on a call
  - The monthly budget is at or above a warning percentage

Usage::

    from driftlock.alerts import WebhookAlertChannel, SlackAlertChannel

    config = DriftlockConfig(
        alert_channels=[
            WebhookAlertChannel(url="https://example.com/hooks/driftlock"),
            SlackAlertChannel(webhook_url="https://hooks.slack.com/services/..."),
        ]
    )

All channels are fire-and-forget.  Delivery failures are logged at WARNING
level and never propagate to the caller.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any, Protocol, runtime_checkable

_log = logging.getLogger("driftlock.alerts")


# ------------------------------------------------------------------ #
# Event types
# ------------------------------------------------------------------ #

ALERT_POLICY_BLOCK = "policy_block"
ALERT_COST_WARNING = "cost_warning"
ALERT_BUDGET_THRESHOLD = "budget_threshold"
ALERT_VELOCITY_TRIP = "velocity_trip"


# ------------------------------------------------------------------ #
# Protocol
# ------------------------------------------------------------------ #

@runtime_checkable
class AlertChannel(Protocol):
    """Any object implementing send() is a valid alert channel."""

    def send(self, event_type: str, payload: dict[str, Any]) -> None:
        """Deliver an alert.  Must not raise — swallow and log failures."""
        ...


# ------------------------------------------------------------------ #
# Implementations
# ------------------------------------------------------------------ #

class WebhookAlertChannel:
    """
    POST a JSON payload to an HTTPS endpoint.

    Payload format::

        {
          "source": "driftlock",
          "event": "policy_block",
          "data": { ... event-specific fields ... }
        }
    """

    def __init__(self, url: str, timeout_seconds: float = 5.0) -> None:
        self._url = url
        self._timeout = timeout_seconds

    def send(self, event_type: str, payload: dict[str, Any]) -> None:
        body = json.dumps(
            {"source": "driftlock", "event": event_type, "data": payload},
            default=str,
        ).encode()
        req = urllib.request.Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout):
                pass
        except Exception as exc:  # noqa: BLE001
            _log.warning("WebhookAlertChannel delivery failed: %s", exc)


class SlackAlertChannel:
    """
    POST a formatted message to a Slack Incoming Webhook URL.

    The message uses Slack's simple text format.  No dependencies beyond stdlib.
    """

    _ICONS = {
        ALERT_POLICY_BLOCK: ":no_entry:",
        ALERT_COST_WARNING: ":warning:",
        ALERT_BUDGET_THRESHOLD: ":money_with_wings:",
        ALERT_VELOCITY_TRIP: ":rotating_light:",
    }

    def __init__(self, webhook_url: str, timeout_seconds: float = 5.0) -> None:
        self._url = webhook_url
        self._timeout = timeout_seconds

    def _format(self, event_type: str, payload: dict[str, Any]) -> str:
        icon = self._ICONS.get(event_type, ":bell:")
        lines = [f"{icon} *Driftlock alert: {event_type}*"]
        for k, v in payload.items():
            lines.append(f"  • *{k}*: `{v}`")
        return "\n".join(lines)

    def send(self, event_type: str, payload: dict[str, Any]) -> None:
        text = self._format(event_type, payload)
        body = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            self._url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout):
                pass
        except Exception as exc:  # noqa: BLE001
            _log.warning("SlackAlertChannel delivery failed: %s", exc)


class LogAlertChannel:
    """
    Writes alerts to the Python logging system.

    Useful for local development, testing, or piping into log aggregators.
    """

    def __init__(self, level: int = logging.WARNING) -> None:
        self._level = level

    def send(self, event_type: str, payload: dict[str, Any]) -> None:
        _log.log(
            self._level,
            "driftlock alert [%s]: %s",
            event_type,
            json.dumps(payload, default=str),
        )


# ------------------------------------------------------------------ #
# Dispatcher helper used by clients
# ------------------------------------------------------------------ #

def fire_alert(
    channels: list[AlertChannel],
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """Fire an alert to all channels.  Never raises."""
    for ch in channels:
        try:
            ch.send(event_type, payload)
        except Exception as exc:  # noqa: BLE001
            _log.warning("Alert channel %s raised: %s", type(ch).__name__, exc)
