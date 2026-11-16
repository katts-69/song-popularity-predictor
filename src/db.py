"""Load the Spotify Tracks CSV into a SQLite database.

Run from the project folder:
    python -m src.db
"""

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


if __name__ == "__main__":
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    n = load_csv_to_sqlite(RAW_CSV, DB_PATH)
    print(f"Loaded {n:,} rows from {RAW_CSV} into {DB_PATH} (table tracks_raw)")
