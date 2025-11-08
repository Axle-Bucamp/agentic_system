from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_settings_round_trip():
    initial = client.get("/api/settings")
    assert initial.status_code == 200
    body = initial.json()
    assert "settings" in body

    update_payload = {"memory_prune_limit": 150, "schedule_profile": "hours"}
    update = client.post("/api/settings", json=update_payload)
    assert update.status_code == 200
    assert update.json()["settings"]["memory_prune_limit"] == 150

    refreshed = client.get("/api/settings")
    assert refreshed.status_code == 200
    assert refreshed.json()["settings"]["schedule_profile"] == "hours"


def test_review_trigger_endpoint():
    response = client.post("/api/review/run")
    assert response.status_code == 200
    assert response.json()["queued"] is True
