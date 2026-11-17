"""Shared test data. pytest loads this file automatically."""

import numpy as np
import pandas as pd
import pytest

GENRE_EFFECT = {"pop": 60, "rock": 40, "jazz": 15}


def make_songs(n, seed):
    """Made-up songs where popularity really depends on genre and energy,
    so a working model has something to learn."""
    rng = np.random.default_rng(seed)
    genres = rng.choice(list(GENRE_EFFECT), size=n)
    energy = rng.uniform(0, 1, size=n)
    df = pd.DataFrame({
        "track_id": [f"t{seed}-{i}" for i in range(n)],
        "track_genre": genres,
        "danceability": rng.uniform(0, 1, n),
        "energy": energy,
        "loudness": rng.uniform(-20, 0, n),
        "speechiness": rng.uniform(0, 0.5, n),
        "acousticness": rng.uniform(0, 1, n),
        "instrumentalness": rng.uniform(0, 1, n),
        "liveness": rng.uniform(0, 1, n),
        "valence": rng.uniform(0, 1, n),
        "tempo": rng.uniform(60, 200, n),
        "duration_ms": rng.integers(120_000, 300_000, n),
        "explicit": rng.integers(0, 2, n),
        "key": rng.integers(0, 12, n),
        "mode": rng.integers(0, 2, n),
        "time_signature": rng.choice([3, 4], n),
    })
    noise = rng.normal(0, 5, n)
    df["popularity"] = np.clip([GENRE_EFFECT[g] for g in genres] - 20 * (energy > 0.8) + noise, 1, 100).round()
    return df


@pytest.fixture(scope="session")
def train_songs():
    return make_songs(600, seed=1)


@pytest.fixture(scope="session")
def unseen_songs():
    return make_songs(200, seed=2)


@pytest.fixture
def one_song():
    """One song as the app would send it: a plain dict of the features."""
    return {
        "track_genre": "jazz", "danceability": 0.5, "energy": 0.4, "loudness": -8.0,
        "speechiness": 0.05, "acousticness": 0.6, "instrumentalness": 0.3, "liveness": 0.1,
        "valence": 0.5, "tempo": 110.0, "duration_ms": 210_000, "explicit": 0,
        "key": 5, "mode": 1, "time_signature": 4,
    }


@pytest.fixture(scope="session")
def small_params():
    """A small, fast model for tests (the real one uses 600 trees)."""
    return {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 15}
