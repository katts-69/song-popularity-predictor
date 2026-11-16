"""Tests for build_clean_table in src/db.py.

Written before the cleaning code exists. Each test is one of the cleaning
decisions made after exploring the data in notebooks/01_sql_exploration.ipynb:

1. Same song ID under several genres -> one row, genre picked at random
   (repeatable), popularity = rounded average of the copies.
2. Same name + artist under different IDs -> keep only the most popular release.
3. Rows missing a song name or artist -> dropped.
4. Popularity 0 -> treated as "no score" and dropped.
"""

import sqlite3

import pandas as pd
import pytest

from src.db import build_clean_table

COLUMNS = ["track_id", "artists", "album_name", "track_name", "popularity", "energy", "track_genre"]


def make_db(tmp_path, rows):
    """Write the given rows into a fresh database as tracks_raw."""
    db_path = tmp_path / "test.db"
    df = pd.DataFrame(rows, columns=COLUMNS)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("tracks_raw", conn, index=False)
    return db_path


def read_clean(db_path):
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql("SELECT * FROM tracks_clean", conn)


def song(track_id, popularity=50, genre="pop", name=None, artist="Artist", album="Album", energy=0.5):
    return [track_id, artist, album, name or f"Song {track_id}", popularity, energy, genre]


# Decision 1: one row per song ID ------------------------------------------------

def test_song_listed_under_three_genres_becomes_one_row(tmp_path):
    db = make_db(tmp_path, [song("a", genre=g) for g in ["blues", "folk", "rock"]])
    build_clean_table(db)
    assert len(read_clean(db)) == 1


def test_chosen_genre_is_one_the_song_really_had(tmp_path):
    db = make_db(tmp_path, [song("a", genre=g) for g in ["blues", "folk", "rock"]])
    build_clean_table(db)
    assert read_clean(db)["track_genre"].iloc[0] in {"blues", "folk", "rock"}


def test_genre_choice_is_not_always_the_alphabetically_first(tmp_path):
    # 30 songs, each listed under the same three genres. If the code just kept
    # the first copy, every song would end up as "a-genre".
    rows = [song(f"s{i}", genre=g) for i in range(30) for g in ["a-genre", "b-genre", "c-genre"]]
    db = make_db(tmp_path, rows)
    build_clean_table(db)
    assert read_clean(db)["track_genre"].nunique() > 1


def test_genre_choice_is_the_same_every_run(tmp_path):
    rows = [song(f"s{i}", genre=g) for i in range(30) for g in ["a-genre", "b-genre", "c-genre"]]
    db = make_db(tmp_path, rows)
    build_clean_table(db)
    first = read_clean(db).sort_values("track_id")["track_genre"].tolist()
    build_clean_table(db)
    second = read_clean(db).sort_values("track_id")["track_genre"].tolist()
    assert first == second


def test_popularity_is_the_rounded_average_of_the_copies(tmp_path):
    db = make_db(tmp_path, [song("a", popularity=40, genre="pop"), song("a", popularity=46, genre="dance")])
    build_clean_table(db)
    assert read_clean(db)["popularity"].iloc[0] == 43


# Decision 2: most popular release of each name + artist -------------------------

def test_only_the_most_popular_release_is_kept(tmp_path):
    db = make_db(tmp_path, [
        song("album", popularity=67, name="93 Million Miles", artist="Jason Mraz", album="Love Is a Four Letter Word"),
        song("compilation", popularity=12, name="93 Million Miles", artist="Jason Mraz", album="Coffee Moment"),
    ])
    build_clean_table(db)
    clean = read_clean(db)
    assert clean["track_id"].tolist() == ["album"]


def test_same_name_by_different_artists_are_different_songs(tmp_path):
    db = make_db(tmp_path, [
        song("x", name="Hello", artist="Adele"),
        song("y", name="Hello", artist="Lionel Richie"),
    ])
    build_clean_table(db)
    assert len(read_clean(db)) == 2


# Decision 3: rows missing a name or artist --------------------------------------

def test_rows_missing_name_or_artist_are_dropped(tmp_path):
    db = make_db(tmp_path, [
        song("good"),
        ["no-name", "Artist", "Album", None, 50, 0.5, "pop"],
        ["no-artist", None, "Album", "Song", 50, 0.5, "pop"],
    ])
    build_clean_table(db)
    assert read_clean(db)["track_id"].tolist() == ["good"]


# Decision 4: popularity 0 means "no score" --------------------------------------

def test_songs_with_popularity_zero_are_dropped(tmp_path):
    db = make_db(tmp_path, [song("scored", popularity=1), song("unscored", popularity=0)])
    build_clean_table(db)
    assert read_clean(db)["track_id"].tolist() == ["scored"]


# General behaviour --------------------------------------------------------------

def test_returns_the_number_of_clean_rows(tmp_path):
    db = make_db(tmp_path, [song("a"), song("b"), song("c", popularity=0)])
    assert build_clean_table(db) == 2


def test_running_twice_does_not_double_the_rows(tmp_path):
    db = make_db(tmp_path, [song("a"), song("b")])
    build_clean_table(db)
    build_clean_table(db)
    assert len(read_clean(db)) == 2


def test_keeps_every_original_column_and_nothing_extra(tmp_path):
    db = make_db(tmp_path, [song("a")])
    build_clean_table(db)
    assert list(read_clean(db).columns) == COLUMNS


def test_raw_table_is_left_untouched(tmp_path):
    rows = [song("a", genre=g) for g in ["blues", "folk"]] + [song("z", popularity=0)]
    db = make_db(tmp_path, rows)
    build_clean_table(db)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tracks_raw").fetchone()[0] == 3
