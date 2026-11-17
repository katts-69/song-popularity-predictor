"""Tests for the security review's fixes. Written before the fixes exist.

1. Rate limit: CORS only stops *browsers* on other websites. A script can call
   the API directly, so each visitor gets a limited number of predictions.
2. Size cap: a song is under 1 KB, so huge request bodies are refused before
   they're read into the server's small memory.
3. GitHub workflows run with read-only permissions.
"""

from pathlib import Path

import pytest

from api.app import create_app
from src.train import train_model

ROOT = Path(__file__).resolve().parent.parent


class CountingStore:
    def __init__(self):
        self.saved = 0

    def save(self, song, prediction):
        self.saved += 1

    def recent(self, limit):
        return []


@pytest.fixture(scope="module")
def bundle(train_songs, small_params):
    return train_model(train_songs, **small_params)


@pytest.fixture
def store():
    return CountingStore()


@pytest.fixture
def client(bundle, store):
    app = create_app(bundle=bundle, store=store, allowed_origins=[], predict_limit="3 per minute")
    return app.test_client()


def predict_from(client, song, visitor_ip):
    # On Render, the visitor's address arrives in X-Forwarded-For (the proxy adds it).
    return client.post("/predict", json=song, headers={"X-Forwarded-For": visitor_ip})


# 1. Rate limit ----------------------------------------------------------------

def test_a_visitor_over_the_limit_is_told_to_wait(client, one_song):
    for _ in range(3):
        assert predict_from(client, one_song, "203.0.113.5").status_code == 200
    response = predict_from(client, one_song, "203.0.113.5")
    assert response.status_code == 429
    assert "wait" in response.get_json()["error"].lower()


def test_refused_requests_are_not_saved(client, store, one_song):
    for _ in range(5):
        predict_from(client, one_song, "203.0.113.5")
    assert store.saved == 3


def test_each_visitor_has_their_own_limit(client, one_song):
    for _ in range(3):
        predict_from(client, one_song, "203.0.113.5")
    assert predict_from(client, one_song, "203.0.113.5").status_code == 429
    assert predict_from(client, one_song, "198.51.100.7").status_code == 200


def test_health_checks_are_never_limited(client):
    # Render calls /health every few seconds; limiting it would mark the server as down.
    for _ in range(20):
        assert client.get("/health", headers={"X-Forwarded-For": "203.0.113.5"}).status_code == 200


def test_reading_genres_and_recent_is_not_limited(client):
    for _ in range(10):
        assert client.get("/genres", headers={"X-Forwarded-For": "203.0.113.5"}).status_code == 200
        assert client.get("/recent", headers={"X-Forwarded-For": "203.0.113.5"}).status_code == 200


def test_the_live_default_limit_is_ten_per_minute(bundle, one_song):
    client = create_app(bundle=bundle, store=CountingStore(), allowed_origins=[]).test_client()
    codes = [predict_from(client, one_song, "203.0.113.9").status_code for _ in range(11)]
    assert codes == [200] * 10 + [429]


# 2. Size cap --------------------------------------------------------------------

def test_huge_request_bodies_are_refused(client, store, one_song):
    padded = {**one_song, "padding": "x" * 20_000}
    response = client.post("/predict", json=padded)
    assert response.status_code == 413
    assert "too large" in response.get_json()["error"].lower()
    assert store.saved == 0


def test_a_normal_song_is_well_under_the_cap(client, one_song):
    assert client.post("/predict", json=one_song).status_code == 200


# 3. GitHub workflows -----------------------------------------------------------

@pytest.mark.parametrize("workflow", ["tests.yml"])
def test_workflows_only_get_read_access(workflow):
    text = (ROOT / ".github" / "workflows" / workflow).read_text()
    assert "permissions:\n  contents: read" in text
