"""Tests for api/app.py: the Flask API the web page talks to.

Written before api/app.py exists. The real API saves predictions to Neon
(PostgreSQL); here a FakeStore stands in for it, so the tests need no
internet and no password.
"""

from datetime import datetime, timezone

import pytest

from api.app import create_app
from src.train import train_model


class FakeStore:
    """Remembers saved predictions in a list instead of a database."""

    def __init__(self, broken=False):
        self.rows = []
        self.broken = broken

    def save(self, song, prediction):
        if self.broken:
            raise ConnectionError("database is down")
        self.rows.append({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "track_genre": song["track_genre"],
            "prediction": prediction,
        })

    def recent(self, limit):
        if self.broken:
            raise ConnectionError("database is down")
        return list(reversed(self.rows))[:limit]


ALLOWED_ORIGIN = "https://katts-69.github.io"


@pytest.fixture(scope="module")
def bundle(train_songs, small_params):
    return train_model(train_songs, **small_params)


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def client(bundle, store):
    app = create_app(bundle=bundle, store=store, allowed_origins=[ALLOWED_ORIGIN])
    return app.test_client()


# GET /health -------------------------------------------------------------------

def test_health_says_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


# GET /genres -------------------------------------------------------------------

def test_genres_lists_what_the_model_knows(client, bundle):
    response = client.get("/genres")
    assert response.status_code == 200
    assert response.get_json() == {"genres": bundle["genres"]}


# POST /predict: the happy path ----------------------------------------------------

def test_predict_returns_a_score_and_reasons(client, one_song):
    response = client.post("/predict", json=one_song)
    assert response.status_code == 200
    body = response.get_json()
    assert 0 <= body["prediction"] <= 100
    assert len(body["reasons"]) == 5
    assert {"starting_point", "other_features"} <= set(body)


def test_predict_matches_the_model_directly(client, bundle, one_song):
    from src.predict import predict_one

    body = client.post("/predict", json=one_song).get_json()
    assert body["prediction"] == pytest.approx(predict_one(bundle, one_song))


def test_each_prediction_saves_exactly_one_row(client, store, one_song):
    client.post("/predict", json=one_song)
    assert len(store.rows) == 1
    assert store.rows[0]["track_genre"] == one_song["track_genre"]


def test_explicit_accepts_true_and_false(client, one_song):
    response = client.post("/predict", json={**one_song, "explicit": True})
    assert response.status_code == 200


def test_a_broken_database_does_not_break_predictions(bundle, one_song):
    # Saving is a bonus; the visitor should still get their score.
    app = create_app(bundle=bundle, store=FakeStore(broken=True), allowed_origins=[ALLOWED_ORIGIN])
    response = app.test_client().post("/predict", json=one_song)
    assert response.status_code == 200
    assert response.get_json()["saved"] is False


def test_a_working_database_reports_saved(client, one_song):
    assert client.post("/predict", json=one_song).get_json()["saved"] is True


# POST /predict: bad input gets a clear 400 ----------------------------------------

def assert_rejected(response, *words):
    assert response.status_code == 400
    message = response.get_json()["error"]
    for word in words:
        assert word in message


def test_body_that_is_not_json_is_rejected(client):
    assert_rejected(client.post("/predict", data="hello", content_type="text/plain"), "JSON")


def test_missing_field_is_rejected_and_named(client, one_song):
    song = {k: v for k, v in one_song.items() if k != "tempo"}
    assert_rejected(client.post("/predict", json=song), "tempo")


def test_unknown_genre_is_rejected_and_named(client, one_song):
    assert_rejected(client.post("/predict", json={**one_song, "track_genre": "polka"}), "polka")


def test_text_where_a_number_belongs_is_rejected(client, one_song):
    assert_rejected(client.post("/predict", json={**one_song, "energy": "loud"}), "energy")


@pytest.mark.parametrize("field, value", [
    ("energy", 1.5),
    ("danceability", -0.1),
    ("tempo", -10),
    ("duration_ms", 0),
    ("key", 12),
    ("mode", 2),
    ("loudness", 20),
])
def test_values_outside_their_range_are_rejected(client, one_song, field, value):
    assert_rejected(client.post("/predict", json={**one_song, field: value}), field)


def test_rejected_requests_are_not_saved(client, store, one_song):
    client.post("/predict", json={**one_song, "energy": 1.5})
    assert store.rows == []


def test_unexpected_extra_fields_are_ignored(client, one_song):
    response = client.post("/predict", json={**one_song, "song_name": "My Song"})
    assert response.status_code == 200


# GET /recent ---------------------------------------------------------------------

def test_recent_is_empty_before_any_prediction(client):
    response = client.get("/recent")
    assert response.status_code == 200
    assert response.get_json() == {"predictions": []}


def test_recent_shows_newest_first(client, one_song):
    client.post("/predict", json={**one_song, "track_genre": "jazz"})
    client.post("/predict", json={**one_song, "track_genre": "pop"})
    genres = [p["track_genre"] for p in client.get("/recent").get_json()["predictions"]]
    assert genres == ["pop", "jazz"]


def test_recent_respects_the_limit(client, one_song):
    for _ in range(5):
        client.post("/predict", json=one_song)
    assert len(client.get("/recent?limit=3").get_json()["predictions"]) == 3


def test_recent_never_returns_more_than_50(client, store, one_song):
    store.rows = [{"created_at": "x", "track_genre": "pop", "prediction": 50.0}] * 80
    assert len(client.get("/recent?limit=1000").get_json()["predictions"]) == 50


def test_recent_rejects_a_limit_that_is_not_a_number(client):
    assert_rejected(client.get("/recent?limit=ten"), "limit")


def test_recent_with_a_broken_database_says_so(bundle):
    app = create_app(bundle=bundle, store=FakeStore(broken=True), allowed_origins=[ALLOWED_ORIGIN])
    response = app.test_client().get("/recent")
    assert response.status_code == 503
    assert "error" in response.get_json()


# CORS: which websites may call the API ----------------------------------------------

def test_our_web_page_is_allowed_to_call_the_api(client):
    response = client.get("/health", headers={"Origin": ALLOWED_ORIGIN})
    assert response.headers.get("Access-Control-Allow-Origin") == ALLOWED_ORIGIN


def test_other_websites_are_not_allowed(client):
    response = client.get("/health", headers={"Origin": "https://some-other-site.com"})
    assert "Access-Control-Allow-Origin" not in response.headers
