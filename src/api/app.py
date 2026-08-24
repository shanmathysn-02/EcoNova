import yaml
from flask import Flask
from flask_cors import CORS
from src.api.routes import api_bp
from src.api.middleware import register_middleware
from src.utils.logger import logger

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_app(config_path: str = "config.yaml") -> Flask:
    """
    Application factory pattern to create and configure the Flask app.
    """
    app = Flask(__name__)
    
    # Load Configuration
    try:
        config = load_config(config_path)
        app.config.update(config)
        logger.info(f"Successfully loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

    # Configure CORS (allow all for development, restrict in production)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Register Middleware (Request IDs, Error Handlers)
    register_middleware(app)
    
    # Register Routes
    app.register_blueprint(api_bp)
    logger.info("Registered API blueprints")
    
    return app
