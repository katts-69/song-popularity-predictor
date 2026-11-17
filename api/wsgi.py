"""The app object servers start: `flask --app api.wsgi run` locally."""

from dotenv import load_dotenv

load_dotenv()  # reads DATABASE_URL and ALLOWED_ORIGINS from .env, if there is one

from api.app import create_app  # noqa: E402  (must come after load_dotenv)

app = create_app()
