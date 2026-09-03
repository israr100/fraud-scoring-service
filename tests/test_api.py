"""API tests using FastAPI's TestClient (spins the app up in-process)."""
import pytest
from fastapi.testclient import TestClient

from src.data import FEATURE_COLUMNS, make_synthetic


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    # Isolate DB + model artifact to a temp dir so tests never touch real state.
    tmp = tmp_path_factory.mktemp("svc")
    monkeypatch_module.setenv("FRAUD_DATABASE_URL", f"sqlite:///{tmp/'test.db'}")
    monkeypatch_module.setenv("FRAUD_MODEL_PATH", str(tmp / "model.joblib"))

    # Reload config + modules so the env vars take effect.
    import importlib
    import src.config as config
    importlib.reload(config)
    import src.db as db
    importlib.reload(db)
    import src.train as train
    importlib.reload(train)
    train.settings = config.settings
    train.train()  # produce an artifact at the temp path

    import src.api as api
    importlib.reload(api)
    with TestClient(api.app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def _example_txn(fraud: bool = False):
    df = make_synthetic(n_rows=2_000, fraud_rate=0.05, seed=7)
    row = df[df["Class"] == (1 if fraud else 0)].iloc[0]
    return {c: float(row[c]) for c in FEATURE_COLUMNS}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_score_returns_valid_probability(client):
    r = client.post("/score", json=_example_txn())
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)


def test_unknown_field_is_rejected(client):
    bad = _example_txn()
    bad["surprise"] = 1.0
    r = client.post("/score", json=bad)
    assert r.status_code == 422  # pydantic extra="forbid"


def test_batch_and_metrics(client):
    batch = [_example_txn(), _example_txn(fraud=True)]
    r = client.post("/score/batch", json=batch)
    assert r.status_code == 200 and len(r.json()) == 2
    m = client.get("/metrics").json()
    assert m["predictions_served"] >= 2
    assert "roc_auc" in m["model_metrics"]
