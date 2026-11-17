"""Tests for src/predict.py: scoring and explaining one song, as the API will.

Written before src/predict.py exists.
"""

import pandas as pd
import pytest

from src.data import get_xy
from src.predict import explain_one, predict_one
from src.train import train_model


@pytest.fixture(scope="module")
def bundle(train_songs, small_params):
    return train_model(train_songs, **small_params)


# Predicting ------------------------------------------------------------------

def test_prediction_is_a_number_from_0_to_100(bundle, one_song):
    score = predict_one(bundle, one_song)
    assert isinstance(score, float)
    assert 0 <= score <= 100


def test_one_song_alone_gets_the_same_score_as_in_a_crowd(bundle, one_song, unseen_songs):
    # A table with one jazz song only knows one genre. If genre labels were
    # numbered per table, "jazz" could be read as the wrong genre.
    crowd = pd.concat([unseen_songs, pd.DataFrame([one_song])], ignore_index=True)
    X, _ = get_xy(crowd.assign(popularity=0), genres=bundle["genres"])
    in_crowd = bundle["model"].predict(X)[-1]
    assert predict_one(bundle, one_song) == pytest.approx(in_crowd)


def test_genre_matters_to_the_prediction(bundle, one_song):
    pop = predict_one(bundle, {**one_song, "track_genre": "pop"})
    jazz = predict_one(bundle, {**one_song, "track_genre": "jazz"})
    assert pop > jazz + 20  # in the made-up data pop is 45 points above jazz


class FakeModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return [self.value] * len(X)


@pytest.mark.parametrize("raw, expected", [(130.0, 100.0), (-5.0, 0.0)])
def test_scores_outside_0_100_are_clipped(bundle, one_song, raw, expected):
    fake = {**bundle, "model": FakeModel(raw)}
    assert predict_one(fake, one_song) == expected


def test_unknown_genre_is_refused_with_a_clear_message(bundle, one_song):
    with pytest.raises(ValueError, match="polka"):
        predict_one(bundle, {**one_song, "track_genre": "polka"})


def test_missing_feature_is_refused_and_named(bundle, one_song):
    song = {k: v for k, v in one_song.items() if k != "energy"}
    with pytest.raises(ValueError, match="energy"):
        predict_one(bundle, song)


# Explaining ------------------------------------------------------------------

def test_explanation_adds_up_to_the_prediction(bundle, one_song):
    result = explain_one(bundle, one_song)
    total = result["starting_point"] + sum(r["push"] for r in result["reasons"]) + result["other_features"]
    assert total == pytest.approx(result["raw_prediction"], abs=1e-6)
    assert result["prediction"] == predict_one(bundle, one_song)


def test_explanation_lists_the_biggest_pushes_first(bundle, one_song):
    reasons = explain_one(bundle, one_song, top=5)["reasons"]
    assert len(reasons) == 5
    sizes = [abs(r["push"]) for r in reasons]
    assert sizes == sorted(sizes, reverse=True)


def test_each_reason_names_the_feature_and_its_value(bundle, one_song):
    reasons = explain_one(bundle, one_song)["reasons"]
    for r in reasons:
        assert set(r) == {"feature", "value", "push"}
        assert r["value"] == one_song[r["feature"]]


def test_genre_is_the_top_reason_when_genre_drives_popularity(bundle, one_song):
    assert explain_one(bundle, one_song)["reasons"][0]["feature"] == "track_genre"
