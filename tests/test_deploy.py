"""Tests for the deployment files. Written before requirements-api.txt and render.yaml exist.

The server installs a slimmer list of libraries (requirements-api.txt) than the
laptop (requirements.txt, which also has Jupyter, SHAP and matplotlib). Two
things must stay true or the live API breaks.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pins(path):
    """{package name: version} for every 'name==version' line in a requirements file."""
    found = {}
    for line in (ROOT / path).read_text().splitlines():
        match = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?==([^\s#]+)", line.strip())
        if match:
            found[match.group(1).lower()] = match.group(3)
    return found


def test_server_uses_the_same_versions_as_the_laptop():
    # The saved model was made with the laptop's lightgbm and scikit-learn.
    # Loading it with different versions can fail or change predictions.
    laptop, server = pins("requirements.txt"), pins("requirements-api.txt")
    shared = set(laptop) & set(server)
    assert {"lightgbm", "scikit-learn", "pandas", "numpy", "flask"} <= shared
    for package in shared:
        assert server[package] == laptop[package], package


def test_server_list_has_a_production_web_server():
    # `flask run` is for development; Render runs the app with gunicorn.
    assert "gunicorn" in pins("requirements-api.txt")


def test_server_list_leaves_out_notebook_only_libraries():
    server = pins("requirements-api.txt")
    for package in ["jupyter", "shap", "matplotlib"]:
        assert package not in server


def test_api_starts_without_notebook_only_libraries():
    # Import the API in a fresh Python where jupyter, shap and matplotlib are
    # blocked, proving the server list is enough.
    code = (
        "import sys\n"
        "for name in ['shap', 'matplotlib', 'jupyter', 'IPython']:\n"
        "    sys.modules[name] = None\n"
        "from api.app import create_app\n"
        "app = create_app(store=None, allowed_origins=[])\n"
        "print(app.test_client().get('/health').status_code)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.stdout.strip() == "200", result.stderr


def test_render_blueprint_starts_the_api_and_checks_health():
    blueprint = (ROOT / "render.yaml").read_text()
    assert "gunicorn api.wsgi:app" in blueprint
    assert "requirements-api.txt" in blueprint
    assert "healthCheckPath: /health" in blueprint
    assert "region: singapore" in blueprint  # same region as the Neon database
