import pytest
from datetime import datetime, timezone

from driftlock.metrics import CallMetrics
from driftlock.storage import SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    return SQLiteStorage(db_path=str(tmp_path / "test.db"))


def _make_metrics(**kwargs) -> CallMetrics:
    defaults = dict(
        model="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        latency_ms=312.5,
        estimated_cost_usd=0.0000225,
        endpoint="test",
        labels={},
        warnings=[],
    )
    defaults.update(kwargs)
    return CallMetrics(**defaults)


def test_save_and_recent(storage):
    storage.save(_make_metrics())
    rows = storage.recent(limit=5)
    assert len(rows) == 1
    assert rows[0]["model"] == "gpt-4o-mini"
    assert rows[0]["prompt_tokens"] == 100


def test_aggregate(storage):
    for _ in range(5):
        storage.save(_make_metrics(prompt_tokens=100, completion_tokens=50))

    agg = storage.aggregate()
    assert agg["calls"] == 5
    assert agg["total_prompt_tokens"] == 500
    assert agg["total_completion_tokens"] == 250


def test_aggregate_by_endpoint(storage):
    storage.save(_make_metrics(endpoint="a", prompt_tokens=10))
    storage.save(_make_metrics(endpoint="b", prompt_tokens=20))

    assert storage.aggregate(endpoint="a")["total_prompt_tokens"] == 10
    assert storage.aggregate(endpoint="b")["total_prompt_tokens"] == 20


def test_aggregate_empty(storage):
    agg = storage.aggregate()
    assert agg["calls"] == 0
    assert agg["total_cost_usd"] == 0
