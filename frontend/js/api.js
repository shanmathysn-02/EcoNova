/**
 * Diabetic Retinopathy Screening System
 * Phase 3 - Centralized API Client Module
 */

// Base URL configuration for backend API (Easy to switch between dev & prod)
const API_BASE_URL = window.API_BASE_URL || 'http://localhost:5000';

/**
 * Custom Error class for API responses and Network failures
 */
class APIError extends Error {
  constructor(message, errorCode = 'UNKNOWN_ERROR', status = 0, responseData = null) {
    super(message);
    this.name = 'APIError';
    this.errorCode = errorCode;
    this.status = status;
    this.responseData = responseData;
  }
}

/**
 * API Client Object
 */
const APIClient = {
  /**
   * Upload retinal image for AI screening prediction
   * @param {File} file - Retinal image file (.jpg, .jpeg, .png)
   * @param {Object} options - Optional flags (e.g. explain, quality_check)
   * @returns {Promise<Object>} Backend API JSON response payload
   */
  async uploadImage(file, options = {}) {
    const formData = new FormData();
    formData.append('file', file);

    if (options.explain !== undefined) formData.append('explain', options.explain);
    if (options.quality_check !== undefined) formData.append('quality_check', options.quality_check);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
        method: 'POST',
        body: formData,
      });

      let data = null;
      try {
        data = await response.json();
      } catch (parseError) {
        // If non-JSON response returned by backend
        data = null;
      }

      if (!response.ok) {
        let errorCode = 'UNKNOWN_ERROR';
        let errorMessage = 'An error occurred while communicating with the server.';

        if (response.status === 400) {
          errorCode = data?.code || data?.error?.code || 'QUALITY_REJECTED';
          errorMessage = data?.message || data?.error?.message || 'Image quality is too low.';
        } else if (response.status === 422) {
          errorCode = data?.code || data?.error?.code || 'INVALID_IMAGE_FORMAT';
          errorMessage = data?.message || data?.error?.message || 'Invalid image format.';
        } else if (response.status === 500) {
          errorCode = data?.code || data?.error?.code || 'INTERNAL_ERROR';
          errorMessage = data?.message || data?.error?.message || 'The prediction service could not process the image.';
        } else {
          errorCode = data?.code || data?.error?.code || `HTTP_${response.status}`;
          errorMessage = data?.message || data?.error?.message || `Server returned error ${response.status}.`;
        }

        throw new APIError(errorMessage, errorCode, response.status, data);
      }

      return data;

    } catch (error) {
      if (error instanceof APIError) {
        throw error;
      }

      // Handle network connection failure (e.g. Flask backend offline)
      throw new APIError(
        'Unable to connect to the screening server. Please make sure the backend server is running and try again.',
        'NETWORK_ERROR',
        0,
        null
      );
    }
  },

  /**
   * Health check endpoint helper (Prepared for future phases)
   */
  async checkHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/health`);
      if (!response.ok) throw new APIError('Health check failed', 'HEALTH_FAILED', response.status);
      return await response.json();
    } catch (err) {
      if (err instanceof APIError) throw err;
      throw new APIError('Backend service unavailable', 'NETWORK_ERROR', 0);
    }
  },

  /**
   * Model metadata endpoint helper (Prepared for future phases)
   */
  async getModelInfo() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/model-info`);
      if (!response.ok) throw new APIError('Failed to fetch model info', 'MODEL_INFO_FAILED', response.status);
      return await response.json();
    } catch (err) {
      if (err instanceof APIError) throw err;
      throw new APIError('Backend service unavailable', 'NETWORK_ERROR', 0);
    }
  }
};
