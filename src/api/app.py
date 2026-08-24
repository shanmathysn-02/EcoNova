import os
from flask import Flask
from .routes import api_bp


def create_app(config_name=None):
    # Set the Flask root_path to the workspace root directory
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    app = Flask(__name__, root_path=workspace_root)

    # Ensure static directory structure exists on startup
    # === MEMBER 2 STATIC SETUP START
    import os
    static_heatmaps_dir = os.path.join(app.root_path, "static", "heatmaps")
    os.makedirs(static_heatmaps_dir, exist_ok=True)
    # === MEMBER 2 STATIC SETUP END

    app.register_blueprint(api_bp)

    return app
