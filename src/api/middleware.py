import uuid
from flask import Flask, jsonify, request, g
from werkzeug.exceptions import HTTPException
from src.utils.exceptions import DRSystemError
from src.utils.logger import logger

def register_middleware(app: Flask):
    
    @app.before_request
    def before_request():
        # Generate a unique request ID for tracing
        g.request_id = uuid.uuid4().hex
        logger.info(f"Received {request.method} request to {request.path}")

    @app.after_request
    def after_request(response):
        # Inject the request ID into the response headers
        if hasattr(g, 'request_id'):
            response.headers['X-Request-ID'] = g.request_id
        return response

    @app.errorhandler(DRSystemError)
    def handle_domain_error(error):
        """Handle custom domain exceptions."""
        logger.error(f"Domain Error [{error.error_code}]: {error.message}")
        response = {
            "success": False,
            "error_code": error.error_code,
            "message": error.message,
            "request_id": getattr(g, 'request_id', 'N/A')
        }
        # We can map specific domain errors to HTTP status codes if needed.
        # For simplicity in Phase 1, using 400 for input, 500 for others.
        status_code = 400 if error.error_code in ["INVALID_INPUT", "QUALITY_REJECTED"] else 500
        return jsonify(response), status_code

    @app.errorhandler(HTTPException)
    def handle_http_error(error):
        """Handle standard Flask HTTP errors (e.g. 404, 405)."""
        logger.warning(f"HTTP Error: {error.code} - {error.name}")
        response = {
            "success": False,
            "error_code": f"HTTP_{error.code}",
            "message": error.description,
            "request_id": getattr(g, 'request_id', 'N/A')
        }
        return jsonify(response), error.code

    @app.errorhandler(Exception)
    def handle_generic_error(error):
        """Handle any other unhandled exceptions safely without exposing stack traces."""
        logger.exception("Unhandled exception occurred")
        response = {
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred. Please try again later.",
            "request_id": getattr(g, 'request_id', 'N/A')
        }
        return jsonify(response), 500
