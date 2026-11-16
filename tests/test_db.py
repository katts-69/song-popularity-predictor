"""Tests for src/db.py: loading the Spotify CSV into a SQLite database.

Written before src/db.py exists. Each test describes one thing the loader
must do, using a tiny made-up CSV so we know the right answer exactly.
"""

import sqlite3

import pandas as pd
import pytest

from src.db import load_csv_to_sqlite


@pytest.fixture
def tiny_csv(tmp_path):
    """A 3-row CSV shaped like the real Kaggle file, including its
    leftover 'Unnamed: 0' row-number column."""
    rows = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "track_id": ["a1", "b2", "c3"],
            "track_name": ["Song A", "Song B", "Song C"],
            "popularity": [73, 55, 0],
            "energy": [0.46, 0.17, 0.91],
            "track_genre": ["acoustic", "acoustic", "rock"],
        }
    )
    path = tmp_path / "tiny.csv"
    rows.to_csv(path, index=False)
    return path


def count_rows(db_path, table):
    with sqlite3.connect(db_path) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def column_names(db_path, table):
    with sqlite3.connect(db_path) as conn:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def test_every_csv_row_ends_up_in_the_table(tiny_csv, tmp_path):
    db_path = tmp_path / "test.db"
    load_csv_to_sqlite(tiny_csv, db_path)
    assert count_rows(db_path, "tracks_raw") == 3


def test_returns_the_number_of_rows_loaded(tiny_csv, tmp_path):
    assert load_csv_to_sqlite(tiny_csv, tmp_path / "test.db") == 3


def test_drops_the_leftover_row_number_column(tiny_csv, tmp_path):
    db_path = tmp_path / "test.db"
    load_csv_to_sqlite(tiny_csv, db_path)
    columns = column_names(db_path, "tracks_raw")
    assert "Unnamed: 0" not in columns
    assert columns == ["track_id", "track_name", "popularity", "energy", "track_genre"]


def test_running_twice_does_not_double_the_rows(tiny_csv, tmp_path):
    db_path = tmp_path / "test.db"
    load_csv_to_sqlite(tiny_csv, db_path)
    load_csv_to_sqlite(tiny_csv, db_path)
    assert count_rows(db_path, "tracks_raw") == 3


def test_numbers_stay_numbers(tiny_csv, tmp_path):
    db_path = tmp_path / "test.db"
    load_csv_to_sqlite(tiny_csv, db_path)
    with sqlite3.connect(db_path) as conn:
        total = conn.execute("SELECT SUM(popularity) FROM tracks_raw").fetchone()[0]
    assert total == 73 + 55 + 0
