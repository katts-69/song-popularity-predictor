"""Train the LightGBM model on the train songs and save it to models/.

Run from the project folder:
    python -m src.train
"""

import json
from pathlib import Path

import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import FEATURES, get_xy, load_clean, split_tracks

DB_PATH = Path("data/spotify.db")
MODEL_PATH = Path("models/model.joblib")
METRICS_PATH = Path("models/metrics.json")

# The settings chosen in notebooks/03_models.ipynb.
DEFAULT_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "random_state": 42,
    "verbose": -1,
}


def train_model(train_df, **params):
    """Train on `train_df` and return a bundle: the model plus the genre list
    and feature order it learned, which predicting needs later."""
    genres = sorted(str(g) for g in train_df["track_genre"].unique())
    X, y = get_xy(train_df, genres=genres)
    model = LGBMRegressor(**{**DEFAULT_PARAMS, **params})
    model.fit(X, y)
    return {"model": model, "genres": genres, "features": list(FEATURES)}


def save_model(bundle, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model(path):
    return joblib.load(path)


def evaluate(bundle, df, average):
    """Scores on `df`, next to the guess-the-average baseline."""
    X, y = get_xy(df, genres=bundle["genres"])
    guess = bundle["model"].predict(X)
    return {
        "mae": round(mean_absolute_error(y, guess), 3),
        "rmse": round(mean_squared_error(y, guess) ** 0.5, 3),
        "r2": round(r2_score(y, guess), 3),
        "baseline_mae": round(mean_absolute_error(y, [average] * len(y)), 3),
        "songs": len(y),
    }


if __name__ == "__main__":
    train, _, test = split_tracks(load_clean(DB_PATH))
    bundle = train_model(train)
    metrics = evaluate(bundle, test, average=train["popularity"].mean())
    save_model(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"Test songs: {metrics}")
