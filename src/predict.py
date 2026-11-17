"""Score and explain one song with a saved model bundle (see src/train.py)."""

import pandas as pd

from src.data import get_x


def _song_to_table(bundle, song):
    """Check the song and turn it into a one-row table the model can read."""
    missing = [f for f in bundle["features"] if f not in song]
    if missing:
        raise ValueError(f"Missing feature(s): {', '.join(missing)}")
    if song["track_genre"] not in bundle["genres"]:
        raise ValueError(f"Unknown genre '{song['track_genre']}'. Choose one of the {len(bundle['genres'])} known genres.")
    row = pd.DataFrame([{f: song[f] for f in bundle["features"]}])
    return get_x(row, genres=bundle["genres"])


def _clip(score):
    return float(min(100.0, max(0.0, score)))


def predict_one(bundle, song):
    """Predicted popularity for one song (a dict of features), from 0 to 100."""
    X = _song_to_table(bundle, song)
    return _clip(bundle["model"].predict(X)[0])


def explain_one(bundle, song, top=5):
    """The prediction split into pushes, biggest first.

    LightGBM computes SHAP values itself (`pred_contrib=True`): one push per
    feature plus the starting point in the last column. It's the same method
    as the shap library in notebook 04, without needing that library here.
    """
    X = _song_to_table(bundle, song)
    contributions = bundle["model"].predict(X, pred_contrib=True)[0]
    starting_point = float(contributions[-1])
    pushes = sorted(
        zip(bundle["features"], (float(p) for p in contributions[:-1])),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )
    reasons = [{"feature": f, "value": song[f], "push": p} for f, p in pushes[:top]]
    other = sum(p for _, p in pushes[top:])
    return {
        "prediction": predict_one(bundle, song),
        "raw_prediction": starting_point + sum(p for _, p in pushes),
        "starting_point": starting_point,
        "reasons": reasons,
        "other_features": other,
    }
