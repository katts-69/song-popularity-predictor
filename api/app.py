"""The Flask API the web page talks to.

Run locally from the project folder:
    flask --app api.wsgi run
"""

import os
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

from src.predict import explain_one
from src.train import load_model

MODEL_PATH = Path("models/model.joblib")
MAX_RECENT = 50
PREDICT_LIMIT = "10 per minute"  # per visitor; CORS can't stop scripts, this can
MAX_BODY_BYTES = 16 * 1024       # a song is under 1 KB

# What each feature is allowed to be: (lowest, highest, must be a whole number).
RANGES = {
    "danceability": (0, 1, False),
    "energy": (0, 1, False),
    "speechiness": (0, 1, False),
    "acousticness": (0, 1, False),
    "instrumentalness": (0, 1, False),
    "liveness": (0, 1, False),
    "valence": (0, 1, False),
    "loudness": (-60, 5, False),
    "tempo": (0, 300, False),
    "duration_ms": (1_000, 6_000_000, True),
    "explicit": (0, 1, True),
    "key": (0, 11, True),
    "mode": (0, 1, True),
    "time_signature": (0, 7, True),
}


class BadInput(ValueError):
    """Something wrong with what the visitor sent: becomes a 400."""


def check_song(body, genres):
    """Return a clean copy of the song, or raise BadInput saying what's wrong."""
    if not isinstance(body, dict):
        raise BadInput("Send the song as a JSON object.")

    missing = [f for f in ["track_genre", *RANGES] if f not in body]
    if missing:
        raise BadInput(f"Missing field(s): {', '.join(missing)}")

    genre = body["track_genre"]
    if genre not in genres:
        raise BadInput(f"Unknown genre '{genre}'. GET /genres lists the {len(genres)} known genres.")

    song = {"track_genre": genre}
    for field, (low, high, whole) in RANGES.items():
        value = body[field]
        if field == "explicit" and isinstance(value, bool):
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BadInput(f"'{field}' must be a number.")
        if whole and float(value) != int(value):
            raise BadInput(f"'{field}' must be a whole number.")
        if not low <= value <= high:
            raise BadInput(f"'{field}' must be between {low} and {high}, got {value}.")
        song[field] = int(value) if whole else float(value)
    return song


def default_store():
    """The Neon database when DATABASE_URL is set, otherwise no saving."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        return None
    from api.store import PostgresStore

    return PostgresStore(url)


def create_app(bundle=None, store="default", allowed_origins=None, predict_limit=PREDICT_LIMIT):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES  # bigger bodies get a 413, unread
    # On Render every request arrives through one proxy, which puts the visitor's
    # real address in X-Forwarded-For. Trust exactly that one hop.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1)
    limiter = Limiter(get_remote_address, app=app, storage_uri="memory://")
    bundle = bundle or load_model(MODEL_PATH)
    store = default_store() if store == "default" else store
    if allowed_origins is None:
        allowed_origins = [o for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o]
    CORS(app, origins=allowed_origins)

    @app.errorhandler(BadInput)
    def bad_input(error):
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(429)
    def too_many(error):
        return jsonify({"error": "Too many predictions from your connection. Please wait a minute and try again."}), 429

    @app.errorhandler(413)
    def too_large(error):
        return jsonify({"error": "That request is too large. Send only the song's features."}), 413

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/genres")
    def genres():
        return jsonify({"genres": bundle["genres"]})

    @app.post("/predict")
    @limiter.limit(predict_limit)
    def predict():
        body = request.get_json(silent=True)
        if body is None:
            raise BadInput("The request body must be JSON.")
        song = check_song(body, bundle["genres"])
        result = explain_one(bundle, song)
        result.pop("raw_prediction")

        saved = False
        if store is not None:
            try:
                store.save(song, result["prediction"])
                saved = True
            except Exception:  # the score still goes back if saving fails
                app.logger.exception("Could not save prediction")
        return jsonify({**result, "saved": saved})

    @app.get("/recent")
    def recent():
        try:
            limit = int(request.args.get("limit", 10))
        except ValueError:
            raise BadInput("'limit' must be a whole number.")
        limit = max(1, min(limit, MAX_RECENT))

        if store is None:
            return jsonify({"error": "Saving predictions isn't set up on this server."}), 503
        try:
            rows = store.recent(limit)
        except Exception:
            app.logger.exception("Could not read recent predictions")
            return jsonify({"error": "Recent predictions are unavailable right now. Try again later."}), 503
        return jsonify({"predictions": rows[:limit]})

    return app
