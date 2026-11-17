"""Tests for the gaps found in step 11's coverage review.

Each of these behaviours had a planted bug that the whole suite missed.
Most are how the live server reads its settings on Render.
"""

import pytest

from api.app import create_app, default_store
from src.train import evaluate, train_model


@pytest.fixture(scope="module")
def bundle(train_songs, small_params):
    return train_model(train_songs, **small_params)


# Settings the live server reads from Render's environment variables -------------

def test_allowed_websites_are_read_from_the_environment(bundle, monkeypatch):
    # Render sets ALLOWED_ORIGINS; several sites are separated by commas.
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://katts-69.github.io,http://localhost:8000")
    client = create_app(bundle=bundle, store=None).test_client()
    for origin in ["https://katts-69.github.io", "http://localhost:8000"]:
        response = client.get("/health", headers={"Origin": origin})
        assert response.headers.get("Access-Control-Allow-Origin") == origin
    other = client.get("/health", headers={"Origin": "https://example.com"})
    assert "Access-Control-Allow-Origin" not in other.headers


def test_no_allowed_websites_setting_means_no_website_is_allowed(bundle, monkeypatch):
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    client = create_app(bundle=bundle, store=None).test_client()
    response = client.get("/health", headers={"Origin": "https://katts-69.github.io"})
    assert "Access-Control-Allow-Origin" not in response.headers


def test_database_url_turns_on_saving(monkeypatch):
    from api.store import PostgresStore

    monkeypatch.setenv("DATABASE_URL", "postgresql://user@example.invalid/db")
    store = default_store()
    assert isinstance(store, PostgresStore)
    assert store.database_url == "postgresql://user@example.invalid/db"


def test_no_database_url_means_no_saving(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert default_store() is None


def test_recent_without_a_database_explains_why(bundle):
    response = create_app(bundle=bundle, store=None, allowed_origins=[]).test_client().get("/recent")
    assert response.status_code == 503
    assert "isn't set up" in response.get_json()["error"]


def test_production_entry_point_serves_the_saved_model(monkeypatch):
    # gunicorn starts `api.wsgi:app` on Render (see render.yaml).
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import importlib

    import api.wsgi

    importlib.reload(api.wsgi)
    client = api.wsgi.app.test_client()
    assert client.get("/health").status_code == 200
    assert len(client.get("/genres").get_json()["genres"]) == 114


# Input checks that had no test --------------------------------------------------

def test_a_json_list_instead_of_an_object_is_rejected(bundle):
    client = create_app(bundle=bundle, store=None, allowed_origins=[]).test_client()
    response = client.post("/predict", json=[1, 2, 3])
    assert response.status_code == 400
    assert "JSON object" in response.get_json()["error"]


@pytest.mark.parametrize("field, value", [("key", 5.5), ("mode", 0.5), ("duration_ms", 200000.5)])
def test_whole_number_fields_refuse_decimals(bundle, one_song, field, value):
    client = create_app(bundle=bundle, store=None, allowed_origins=[]).test_client()
    response = client.post("/predict", json={**one_song, field: value})
    assert response.status_code == 400
    assert field in response.get_json()["error"]


# The scores written to models/metrics.json (they go in the README) ---------------

def test_evaluate_reports_the_model_and_the_baseline_separately(bundle, train_songs, unseen_songs):
    metrics = evaluate(bundle, unseen_songs, average=train_songs["popularity"].mean())
    assert set(metrics) == {"mae", "rmse", "r2", "baseline_mae", "songs"}
    assert metrics["songs"] == len(unseen_songs)
    assert metrics["mae"] < metrics["baseline_mae"]
    assert metrics["rmse"] >= metrics["mae"]  # true for any set of errors
