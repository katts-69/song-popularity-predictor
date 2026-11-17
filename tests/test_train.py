"""Tests for src/train.py: training the model and saving it to a file.

Written before src/train.py exists.
"""

from pathlib import Path

import pytest
from sklearn.metrics import mean_absolute_error

from src.data import FEATURES, get_xy
from src.train import load_model, save_model, train_model

REAL_DB = Path("data/spotify.db")


@pytest.fixture(scope="module")
def bundle(train_songs, small_params):
    return train_model(train_songs, **small_params)


def test_trained_model_beats_guessing_the_average(bundle, train_songs, unseen_songs):
    X, y = get_xy(unseen_songs, genres=bundle["genres"])
    model_mae = mean_absolute_error(y, bundle["model"].predict(X))
    baseline_mae = mean_absolute_error(y, [train_songs["popularity"].mean()] * len(y))
    assert model_mae < baseline_mae


def test_bundle_remembers_the_genres_it_learned(bundle):
    assert bundle["genres"] == ["jazz", "pop", "rock"]


def test_bundle_remembers_the_feature_order(bundle):
    assert bundle["features"] == FEATURES


def test_saved_model_predicts_exactly_like_the_original(bundle, unseen_songs, tmp_path):
    path = tmp_path / "model.joblib"
    save_model(bundle, path)
    loaded = load_model(path)
    X, _ = get_xy(unseen_songs, genres=bundle["genres"])
    assert list(loaded["model"].predict(X)) == list(bundle["model"].predict(X))
    assert loaded["genres"] == bundle["genres"]


@pytest.mark.skipif(not REAL_DB.exists(), reason="real data is only on the developer's machine")
def test_real_model_beats_the_baseline_on_the_test_songs():
    from src.data import load_clean, split_tracks

    train, _, test = split_tracks(load_clean(REAL_DB))
    real = train_model(train, n_estimators=100)
    X, y = get_xy(test, genres=real["genres"])
    model_mae = mean_absolute_error(y, real["model"].predict(X))
    baseline_mae = mean_absolute_error(y, [train["popularity"].mean()] * len(y))
    assert model_mae < baseline_mae
