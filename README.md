# Explainable AI-Based Diabetic Retinopathy Screening System

## Project Purpose
This project is an Explainable AI (XAI) system for screening Diabetic Retinopathy (DR) from retinal images. The system is designed to provide transparent, interpretable AI predictions (using Grad-CAM) alongside the diagnostic outcome, making it easier for healthcare professionals to trust and verify the results.

## Backend Role (Phase 1)
The current implementation represents **Phase 1: Backend Foundation**. 
It sets up a clean, modular Flask API foundation that will later connect to the Image Preprocessing module, the AI Model, the Explainability module (Grad-CAM), and the Frontend.

*Note: In Phase 1, the AI model, preprocessing, and Grad-CAM modules are intentionally not integrated.*

## Directory Structure
```
dr-detection-system/
├── src/
│   ├── api/          # Flask application factory, routes, and middleware
│   └── utils/        # Shared utilities (logger, exceptions, image_io)
├── tests/            # Pytest unit tests
├── config.yaml       # Application configuration
├── requirements.txt  # Python dependencies
└── run.py            # Application entry point
```

## Installation
1. Ensure you have Python 3.10+ installed.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
Start the Flask development server using:
```bash
python run.py
```
The API will be available at `http://127.0.0.1:5000`

## Available Endpoints (Phase 1)

### 1. Health Check
`GET /api/v1/health`

Verifies backend connectivity.
**Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": false,
  "model_version": null
}
```

### 2. Model Info
`GET /api/v1/model/info`

Provides information about the AI model (Not loaded in Phase 1).
**Response (200 OK):**
```json
{
  "success": false,
  "message": "Model is not loaded. ML integration will be implemented in Phase 4.",
  "model": null
}
```

## Error Handling & Request IDs
All API responses include an `X-Request-ID` header. In case of an error, the API returns a standardized JSON format:
```json
{
  "success": false,
  "error_code": "ERROR_CODE",
  "message": "Human-readable error message",
  "request_id": "unique-uuid-here"
}
```

## Integrating Future Phases (Phase 2 & Beyond)
- **Phase 2 (Image Quality & Preprocessing):** Future modules will use the validators in `src/api/validators.py` and `src/utils/image_io.py`. A new POST endpoint will be added to `src/api/routes.py` that will receive images, validate them, and pass them to the preprocessing pipeline before returning a response.
- **Phase 3 & 4 (AI Model & Grad-CAM):** The ML models will be loaded dynamically, updating the `/health` and `/model/info` responses. Predictions will be formatted and returned with XAI visualizations.
