"""Load the Spotify Tracks CSV into SQLite, then build a cleaned table from it.

Run from the project folder:
    python -m src.db
"""

import hashlib
import sqlite3
from pathlib import Path

import pandas as pd

RAW_CSV = Path("data/raw/dataset.csv")
DB_PATH = Path("data/spotify.db")


def load_csv_to_sqlite(csv_path, db_path, table="tracks_raw"):
    """Copy every row of the CSV into `table`, replacing any earlier copy.

    Returns the number of rows loaded.
    """
    df = pd.read_csv(csv_path)

    # The Kaggle file was saved with its row numbers as an extra column
    # called "Unnamed: 0". It carries no information, so drop it.
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    with sqlite3.connect(db_path) as conn:
        # "replace" drops the old table first, so rerunning never stacks copies.
        df.to_sql(table, conn, if_exists="replace", index=False)

    return len(df)


def _seeded_rank(seed):
    """Make a SQL function that gives each (song, genre) pair a fixed
    pseudo-random number. Sorting by it shuffles a song's genres the same
    way every run, which SQLite's own RANDOM() can't do."""

    def rank(track_id, genre):
        digest = hashlib.sha256(f"{seed}:{track_id}:{genre}".encode()).hexdigest()
        return int(digest[:12], 16)

    return rank


CLEAN_SQL = """
WITH
-- Rule 3: drop rows with no song name or artist.
named AS (
    SELECT * FROM {source}
    WHERE track_name IS NOT NULL AND artists IS NOT NULL
),

-- Rule 1a: one popularity per song ID, the rounded average of its copies.
avg_pop AS (
    SELECT track_id, CAST(ROUND(AVG(popularity)) AS INTEGER) AS avg_popularity
    FROM named
    GROUP BY track_id
),

-- Rule 1b: number each song's copies 1, 2, 3... in a shuffled but repeatable
-- order. Keeping copy 1 picks one genre at random.
ranked_copies AS (
    SELECT named.*, avg_pop.avg_popularity,
           ROW_NUMBER() OVER (
               PARTITION BY named.track_id
               ORDER BY seeded_rank(named.track_id, named.track_genre)
           ) AS copy_rank
    FROM named
    JOIN avg_pop USING (track_id)
),

-- Rule 2: among releases with the same name and artist, number them from most
-- to least popular. Keeping release 1 keeps the most popular one.
ranked_releases AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY track_name, artists
               ORDER BY avg_popularity DESC, track_id
           ) AS release_rank
    FROM ranked_copies
    WHERE copy_rank = 1
)

-- Rule 4: popularity 0 means Spotify has no score, so drop it.
SELECT {columns}
FROM ranked_releases
WHERE release_rank = 1
  AND avg_popularity > 0
"""


def build_clean_table(db_path, source="tracks_raw", target="tracks_clean", seed=42):
    """Build `target` from `source` using the four cleaning rules in CLEAN_SQL.

    Returns the number of rows in the clean table.
    """
    with sqlite3.connect(db_path) as conn:
        conn.create_function("seeded_rank", 2, _seeded_rank(seed), deterministic=True)

        # Keep the source's columns in their original order, swapping in the
        # averaged popularity.
        names = [row[1] for row in conn.execute(f"PRAGMA table_info({source})")]
        columns = ", ".join(
            "avg_popularity AS popularity" if name == "popularity" else f'"{name}"'
            for name in names
        )

        conn.execute(f"DROP TABLE IF EXISTS {target}")
        conn.execute(f"CREATE TABLE {target} AS " + CLEAN_SQL.format(source=source, columns=columns))
        return conn.execute(f"SELECT COUNT(*) FROM {target}").fetchone()[0]


if __name__ == "__main__":
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    n = load_csv_to_sqlite(RAW_CSV, DB_PATH)
    print(f"Loaded {n:,} rows from {RAW_CSV} into {DB_PATH} (table tracks_raw)")
    n = build_clean_table(DB_PATH)
    print(f"Built tracks_clean with {n:,} songs")
