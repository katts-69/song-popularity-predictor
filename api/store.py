"""Save predictions to PostgreSQL (Neon) and read the latest ones back."""

import re
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SAFE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class PostgresStore:
    def __init__(self, database_url, table="predictions"):
        # The table name is written into the SQL text (values are not: they
        # travel separately as %s parameters), so only allow a plain name.
        if not SAFE_NAME.fullmatch(table):
            raise ValueError(f"Unsafe table name: {table!r}")
        self.database_url = database_url
        self.table = table

    def _connect(self):
        return psycopg.connect(self.database_url, connect_timeout=5)

    def create_table(self):
        """Run api/schema.sql, pointed at this store's table."""
        sql = SCHEMA_PATH.read_text().replace("predictions", self.table)
        with self._connect() as conn:
            conn.execute(sql)

    def save(self, song, prediction):
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO {self.table} (track_genre, features, prediction) VALUES (%s, %s, %s)",
                (song["track_genre"], Jsonb(song), prediction),
            )

    def recent(self, limit):
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT created_at, track_genre, prediction FROM {self.table} "
                "ORDER BY created_at DESC, id DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return [
            {"created_at": created.isoformat(), "track_genre": genre, "prediction": float(score)}
            for created, genre, score in rows
        ]
