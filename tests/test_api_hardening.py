from fastapi.testclient import TestClient

from api.server import app


SCORE_PAYLOAD = {
    "account_id": "acct_001",
    "visits": 18,
    "spend": 42000,
    "account_age_days": 640,
    "usage_frequency": 88,
    "support_tickets": 1,
    "renewal_days": 30,
}
AUTH_HEADERS = {"X-API-Key": "test-key"}


def test_request_id_header_is_returned():
    with TestClient(app) as client:
        response = client.get("/health", headers={"X-Request-ID": "req-123"})

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-123"


def test_api_key_is_optional_and_enforced(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with TestClient(app) as client:
        assert client.post("/score", json=SCORE_PAYLOAD).status_code == 200

    monkeypatch.setenv("API_KEY", "test-key")
    with TestClient(app) as client:
        assert client.post("/score", json=SCORE_PAYLOAD).status_code == 401
        assert client.post("/score", json=SCORE_PAYLOAD, headers=AUTH_HEADERS).status_code == 200


def test_validation_errors_use_safe_json_shape(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with TestClient(app) as client:
        response = client.post("/score", json={"spend": -1})

    assert response.status_code == 422
    assert response.json()["error"] == "Invalid request"


def test_metrics_and_event_persistence(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "events.sqlite3"))

    with TestClient(app) as client:
        score_response = client.post("/score", json=SCORE_PAYLOAD, headers=AUTH_HEADERS)
        metrics_response = client.get("/metrics")
        events_response = client.get("/events", headers=AUTH_HEADERS)

    assert score_response.status_code == 200
    assert metrics_response.status_code == 200
    assert "scores_total" in metrics_response.json()["counters"]
    assert events_response.status_code == 200
    assert events_response.json()["events"][0]["event_type"] == "sales_score"
