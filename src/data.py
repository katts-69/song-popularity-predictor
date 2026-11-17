"""Load the clean songs, split them, and pick the columns the model may see."""

import sqlite3

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "popularity"

# Everything a visitor can enter in the app. No IDs or names: they can't be
# typed in, and letting the model memorise them would be cheating.
FEATURES = [
    "track_genre",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "explicit",
    "key",
    "mode",
    "time_signature",
]


def load_clean(db_path):
    """Read the tracks_clean table built by src/db.py."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql("SELECT * FROM tracks_clean", conn)


def split_tracks(df, seed=42, val_share=0.15, test_share=0.15):
    """Split songs into train / validation / test, keeping each genre's share
    the same in all three ("stratified"). Returns (train, val, test)."""
    n_test = round(len(df) * test_share)
    n_val = round(len(df) * val_share)

    rest, test = train_test_split(
        df, test_size=n_test, stratify=df["track_genre"], random_state=seed
    )
    train, val = train_test_split(
        rest, test_size=n_val, stratify=rest["track_genre"], random_state=seed
    )
    return train, val, test


def get_x(df, genres=None):
    """The model's inputs. Pass `genres` (the list the model learned) so every
    table numbers the genres the same way, even a table with one song."""
    X = df[FEATURES].copy()
    if genres is None:
        X["track_genre"] = X["track_genre"].astype("category")
    else:
        X["track_genre"] = pd.Categorical(X["track_genre"], categories=genres)
    return X


def get_xy(df, genres=None):
    """Split a table into model inputs X and the answer y."""
    return get_x(df, genres), df[TARGET]
