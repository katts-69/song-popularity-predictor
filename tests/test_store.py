"""Tests for api/store.py: saving predictions to PostgreSQL (Neon).

Written before api/store.py exists. Most of these talk to the real database,
so they need DATABASE_URL (from .env). They use a throwaway table with a random
name and delete it afterwards, so the real `predictions` log is never touched.
On GitHub there is no DATABASE_URL, so those tests are skipped there.
"""

import os
import uuid
from datetime import datetime

import psycopg
import pytest
from dotenv import load_dotenv

from api.store import PostgresStore

load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL")
needs_db = pytest.mark.skipif(not DATABASE_URL, reason="needs DATABASE_URL (Neon), only set on the developer's machine")


@pytest.fixture
def store():
    table = f"predictions_test_{uuid.uuid4().hex[:8]}"
    store = PostgresStore(DATABASE_URL, table=table)
    store.create_table()
    yield store
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table}")


# No database needed ------------------------------------------------------------

@pytest.mark.parametrize("bad_name", ["x; DROP TABLE predictions", "my table", "", "1abc"])
def test_unsafe_table_names_are_refused(bad_name):
    # The table name goes into the SQL text, so it must be a plain name.
    with pytest.raises(ValueError):
        PostgresStore("postgresql://unused", table=bad_name)


def test_saving_to_an_unreachable_database_raises(one_song):
    # The API catches this and still returns the score (tested in test_api.py).
    store = PostgresStore("postgresql://nobody@127.0.0.1:1/nothing")
    with pytest.raises(psycopg.OperationalError):
        store.save(one_song, 50.0)


# Real database -------------------------------------------------------------------

@needs_db
def test_new_table_starts_empty(store):
    assert store.recent(10) == []


@needs_db
def test_creating_the_table_twice_is_harmless(store):
    store.create_table()
    assert store.recent(10) == []


@needs_db
def test_saved_prediction_comes_back(store, one_song):
    store.save(one_song, 57.6)
    [row] = store.recent(10)
    assert row["track_genre"] == one_song["track_genre"]
    assert row["prediction"] == pytest.approx(57.6, abs=0.01)


@needs_db
def test_recent_is_newest_first(store, one_song):
    for genre in ["jazz", "rock", "pop"]:
        store.save({**one_song, "track_genre": genre}, 40.0)
    assert [r["track_genre"] for r in store.recent(10)] == ["pop", "rock", "jazz"]


@needs_db
def test_recent_respects_the_limit(store, one_song):
    for _ in range(4):
        store.save(one_song, 40.0)
    assert len(store.recent(2)) == 2


@needs_db
def test_saved_time_is_a_readable_timestamp_with_timezone(store, one_song):
    store.save(one_song, 40.0)
    created = datetime.fromisoformat(store.recent(1)[0]["created_at"])
    assert created.tzinfo is not None


@needs_db
def test_the_whole_song_is_kept_for_later_analysis(store, one_song):
    store.save(one_song, 40.0)
    with psycopg.connect(DATABASE_URL) as conn:
        energy = conn.execute(f"SELECT features->>'energy' FROM {store.table}").fetchone()[0]
    assert float(energy) == one_song["energy"]


@needs_db
def test_api_saves_to_the_real_database_end_to_end(store, one_song, train_songs, small_params):
    from api.app import create_app
    from src.train import train_model

    app = create_app(bundle=train_model(train_songs, **small_params), store=store, allowed_origins=[])
    client = app.test_client()
    body = client.post("/predict", json=one_song).get_json()
    assert body["saved"] is True
    recent = client.get("/recent").get_json()["predictions"]
    assert recent[0]["prediction"] == pytest.approx(body["prediction"], abs=0.01)
