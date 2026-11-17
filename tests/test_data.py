"""Tests for src/data.py: splitting songs into train / validation / test,
and choosing which columns the model may see.

Written before src/data.py exists.
"""

import pandas as pd
import pytest

from src.data import FEATURES, TARGET, get_xy, split_tracks

ID_COLUMNS = ["track_id", "artists", "album_name", "track_name"]


@pytest.fixture
def songs():
    """200 made-up songs over 4 genres of different sizes (100, 50, 30, 20),
    one row per song, like tracks_clean."""
    sizes = {"pop": 100, "rock": 50, "jazz": 30, "iranian": 20}
    rows = []
    for genre, n in sizes.items():
        for i in range(n):
            rows.append({
                "track_id": f"{genre}-{i}",
                "artists": f"Artist {i % 7}",
                "album_name": "Album",
                "track_name": f"Song {i}",
                "popularity": (i * 7) % 100 + 1,
                "duration_ms": 200_000 + i,
                "explicit": i % 2,
                "danceability": 0.5, "energy": 0.5, "key": i % 12, "loudness": -6.0,
                "mode": 1, "speechiness": 0.05, "acousticness": 0.2,
                "instrumentalness": 0.0, "liveness": 0.1, "valence": 0.5,
                "tempo": 120.0, "time_signature": 4,
                "track_genre": genre,
            })
    return pd.DataFrame(rows)


# The split ------------------------------------------------------------------

def test_split_sizes_are_70_15_15(songs):
    train, val, test = split_tracks(songs)
    assert (len(train), len(val), len(test)) == (140, 30, 30)


def test_every_song_is_in_exactly_one_split(songs):
    train, val, test = split_tracks(songs)
    ids = pd.concat([train, val, test])["track_id"]
    assert ids.is_unique
    assert set(ids) == set(songs["track_id"])


def test_no_song_appears_in_both_train_and_test(songs):
    train, val, test = split_tracks(songs)
    assert set(train["track_id"]).isdisjoint(test["track_id"])
    assert set(train["track_id"]).isdisjoint(val["track_id"])
    assert set(val["track_id"]).isdisjoint(test["track_id"])


def test_each_genre_keeps_its_share_in_every_split(songs):
    # "Stratified": a genre that is 10% of all songs is about 10% of each split,
    # so small genres are never missing from the test set by bad luck.
    train, val, test = split_tracks(songs)
    for part in (train, val, test):
        shares = part["track_genre"].value_counts(normalize=True)
        expected = songs["track_genre"].value_counts(normalize=True)
        for genre in expected.index:
            assert shares.get(genre, 0) == pytest.approx(expected[genre], abs=0.04)


def test_same_seed_gives_the_same_split(songs):
    first = split_tracks(songs, seed=1)[2]["track_id"].tolist()
    second = split_tracks(songs, seed=1)[2]["track_id"].tolist()
    assert first == second


def test_different_seed_gives_a_different_split(songs):
    a = set(split_tracks(songs, seed=1)[2]["track_id"])
    b = set(split_tracks(songs, seed=2)[2]["track_id"])
    assert a != b


def test_split_does_not_change_the_input(songs):
    before = songs.copy()
    split_tracks(songs)
    pd.testing.assert_frame_equal(songs, before)


# What the model may see ------------------------------------------------------

def test_target_is_popularity():
    assert TARGET == "popularity"


def test_features_never_include_the_answer():
    assert TARGET not in FEATURES


def test_features_never_include_names_or_ids():
    # A song's ID or name can't be typed into the app, and letting the model
    # memorise names would be cheating.
    assert not set(ID_COLUMNS) & set(FEATURES)


def test_features_include_genre_and_the_audio_measurements():
    assert "track_genre" in FEATURES
    for column in ["danceability", "energy", "loudness", "tempo", "duration_ms", "instrumentalness"]:
        assert column in FEATURES


def test_get_xy_returns_feature_table_and_target(songs):
    X, y = get_xy(songs)
    assert list(X.columns) == FEATURES
    assert y.name == TARGET
    assert len(X) == len(y) == len(songs)


def test_get_xy_marks_genre_as_a_category(songs):
    # Models treat text differently from numbers; genre must be a category.
    X, _ = get_xy(songs)
    assert isinstance(X["track_genre"].dtype, pd.CategoricalDtype)
