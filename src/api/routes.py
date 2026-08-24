from flask import Blueprint, jsonify

api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify backend connectivity.
    Phase 1: Model is not yet integrated.
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": False,
        "model_version": None
    }), 200

@api_bp.route('/model/info', methods=['GET'])
def model_info():
    """
    Information about the currently loaded model.
    Phase 1: Model is not yet loaded.
    """
    return jsonify({
        "success": False,
        "message": "Model is not loaded. ML integration will be implemented in Phase 4.",
        "model": None
    }), 200
