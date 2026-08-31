/**
 * CardioLens AI - Frontend API Client
 * Version 2.1.0
 * 
 * Fixes:
 * - Dual health check endpoints (bypass ad-blockers)
 * - Improved error handling and logging
 * - CORS-compliant credential handling
 * - Resilient fallback logic
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || "https://cardiolens-ai-za8w.onrender.com";

// ============================================================================
// API CLIENT CONFIGURATION
// ============================================================================

/**
 * Health check endpoints in priority order
 * /api/status: Recommended (bypasses ad-blockers)
 * /api/health: Standard (may be blocked by extensions)
 * /health: Legacy fallback
 */
const HEALTH_ENDPOINTS = [
  `${API_BASE_URL}/api/status`,
  `${API_BASE_URL}/api/health`,
  `${API_BASE_URL}/health`,
];

const PREDICTION_ENDPOINTS = {
  predict: `${API_BASE_URL}/api/predict`,
  explain: `${API_BASE_URL}/api/explain`,
  whatIf: `${API_BASE_URL}/api/what-if`,
  batchPredict: `${API_BASE_URL}/api/batch-predict`,
};

// ============================================================================
// HEALTH CHECK (RESILIENT)
// ============================================================================

/**
 * Performs health check with fallback endpoints
 * Handles CORS, ad-blocker interference, and network errors
 * 
 * @returns {Promise<{ok: boolean, status: string, modelLoaded: boolean, endpoint: string}>}
 */
export async function healthCheck() {
  const results = {
    ok: false,
    status: "offline",
    modelLoaded: false,
    endpoint: null,
    errors: [],
  };

  // Try each endpoint in sequence
  for (const url of HEALTH_ENDPOINTS) {
    try {
      console.debug(`🔍 Health check attempt: ${url}`);

      const response = await fetch(url, {
        method: "GET",
        credentials: "include",
        headers: {
          "Accept": "application/json",
          "Content-Type": "application/json",
        },
        // Don't cache health checks
        cache: "no-store",
      });

      if (response.ok) {
        const data = await response.json();
        
        results.ok = true;
        results.status = data.status || "unknown";
        results.modelLoaded = data.model_loaded || false;
        results.endpoint = url;
        
        console.log(`✅ Health check successful (${url}):`, data);
        return results;
      } else {
        const errorMsg = `HTTP ${response.status}`;
        results.errors.push({ endpoint: url, error: errorMsg });
        console.warn(`⚠️  Health check failed at ${url}: ${errorMsg}`);
      }
    } catch (err) {
      const errorMsg = err.message || "Unknown error";
      results.errors.push({ endpoint: url, error: errorMsg });
      
      // Distinguish between different error types
      if (err.message.includes("Failed to fetch")) {
        console.debug(`🔌 Network/CORS issue at ${url}: ${errorMsg}`);
      } else if (err.name === "TypeError") {
        console.debug(`❌ Request error at ${url}: ${errorMsg}`);
      } else {
        console.debug(`⚠️  Error at ${url}: ${errorMsg}`);
      }
    }
  }

  console.error("❌ All health check endpoints failed:", results.errors);
  return results;
}

// ============================================================================
// PREDICTION ENDPOINT
// ============================================================================

/**
 * Send patient data for risk prediction
 * 
 * @param {Object} patientData - Patient clinical parameters
 * @returns {Promise<Object>} Prediction response with risk scores
 */
export async function predictRisk(patientData) {
  if (!patientData || Object.keys(patientData).length === 0) {
    throw new Error("Patient data is required");
  }

  try {
    console.debug("📤 Sending prediction request:", patientData);

    const response = await fetch(PREDICTION_ENDPOINTS.predict, {
      method: "POST",
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
      },
      body: JSON.stringify(patientData),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.detail || `Prediction failed: HTTP ${response.status}`
      );
    }

    const result = await response.json();
    console.log("✅ Prediction received:", result);
    return result;
  } catch (err) {
    console.error("❌ Prediction error:", err);
    throw err;
  }
}

// ============================================================================
// EXPLANATION ENDPOINT (SHAP)
// ============================================================================

/**
 * Get feature importance explanation for prediction
 * 
 * @param {Object} patientData - Patient clinical parameters
 * @returns {Promise<Object>} SHAP-based explanations
 */
export async function explainPrediction(patientData) {
  if (!patientData || Object.keys(patientData).length === 0) {
    throw new Error("Patient data is required");
  }

  try {
    console.debug("📤 Sending explanation request:", patientData);

    const response = await fetch(PREDICTION_ENDPOINTS.explain, {
      method: "POST",
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
      },
      body: JSON.stringify(patientData),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.detail || `Explanation failed: HTTP ${response.status}`
      );
    }

    const result = await response.json();
    console.log("✅ Explanation received:", result);
    return result;
  } catch (err) {
    console.error("❌ Explanation error:", err);
    throw err;
  }
}

// ============================================================================
// WHAT-IF (COUNTERFACTUAL) ENDPOINT
// ============================================================================

/**
 * Compare risk between original and modified patient data
 * 
 * @param {Object} original - Original patient parameters
 * @param {Object} modified - Modified patient parameters
 * @returns {Promise<Object>} Counterfactual comparison results
 */
export async function whatIfAnalysis(original, modified) {
  if (!original || !modified) {
    throw new Error("Both original and modified data are required");
  }

  try {
    console.debug("📤 Sending what-if request:", { original, modified });

    const response = await fetch(PREDICTION_ENDPOINTS.whatIf, {
      method: "POST",
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json",
      },
      body: JSON.stringify({ original, modified }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.detail || `What-if analysis failed: HTTP ${response.status}`
      );
    }

    const result = await response.json();
    console.log("✅ What-if analysis received:", result);
    return result;
  } catch (err) {
    console.error("❌ What-if analysis error:", err);
    throw err;
  }
}

// ============================================================================
// BATCH PREDICTION ENDPOINT
// ============================================================================

/**
 * Submit CSV file for bulk predictions
 * 
 * @param {File} csvFile - CSV file with patient records
 * @returns {Promise<Object>} Batch prediction results
 */
export async function batchPredict(csvFile) {
  if (!csvFile) {
    throw new Error("CSV file is required");
  }

  if (!csvFile.name.toLowerCase().endsWith(".csv")) {
    throw new Error("File must be a CSV file");
  }

  try {
    console.debug("📤 Uploading CSV for batch prediction:", csvFile.name);

    const formData = new FormData();
    formData.append("file", csvFile);

    const response = await fetch(PREDICTION_ENDPOINTS.batchPredict, {
      method: "POST",
      credentials: "include",
      body: formData,
      // Don't set Content-Type header - browser will set it with boundary
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.detail || `Batch prediction failed: HTTP ${response.status}`
      );
    }

    const result = await response.json();
    console.log(
      `✅ Batch prediction received: ${result.total_records} records processed`
    );
    return result;
  } catch (err) {
    console.error("❌ Batch prediction error:", err);
    throw err;
  }
}

// ============================================================================
// ERROR HANDLERS
// ============================================================================

/**
 * Format error message for UI display
 * 
 * @param {Error|string} error - Error object or message
 * @returns {string} Formatted error message
 */
export function formatErrorMessage(error) {
  if (typeof error === "string") {
    return error;
  }

  if (error instanceof Error) {
    return error.message;
  }

  if (error?.detail) {
    return error.detail;
  }

  return "An unexpected error occurred";
}

/**
 * Check if error is due to backend being offline
 * 
 * @param {Error} error - Error object
 * @returns {boolean} True if backend is offline
 */
export function isBackendOffline(error) {
  if (!error) return false;

  const message = error.message || error.toString();
  return (
    message.includes("Failed to fetch") ||
    message.includes("Network") ||
    message.includes("CORS") ||
    message.includes("offline")
  );
}

/**
 * Check if error is due to model not being loaded
 * 
 * @param {Error} error - Error object
 * @returns {boolean} True if model is not available
 */
export function isModelUnavailable(error) {
  if (!error) return false;

  const message = error.message || error.toString();
  return (
    message.includes("Model not available") ||
    message.includes("503") ||
    message.includes("degraded")
  );
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Get human-readable status based on health check result
 * 
 * @param {Object} healthStatus - Result from healthCheck()
 * @returns {Object} Formatted status info
 */
export function getStatusDisplay(healthStatus) {
  if (healthStatus.ok && healthStatus.modelLoaded) {
    return {
      label: "✅ Ready",
      color: "green",
      description: "Backend online with model loaded",
    };
  }

  if (healthStatus.ok && !healthStatus.modelLoaded) {
    return {
      label: "⚠️  Degraded",
      color: "yellow",
      description: "Backend online but model not loaded (predictions unavailable)",
    };
  }

  return {
    label: "❌ Offline",
    color: "red",
    description: "Backend unreachable (CORS blocked or server down)",
  };
}

/**
 * Format prediction response for UI display
 * 
 * @param {Object} prediction - Prediction response
 * @returns {Object} Formatted prediction
 */
export function formatPrediction(prediction) {
  return {
    riskPercentage: (prediction.risk_percentage || 0).toFixed(1),
    riskCategory: prediction.risk_category || "Unknown",
    bmi: (prediction.bmi || 0).toFixed(1),
    disclaimer: prediction.disclaimer || "",
  };
}

// ============================================================================
// REACT HOOK EXAMPLE
// ============================================================================

/**
 * React hook for managing backend connection state
 * 
 * Usage in component:
 * const [backendStatus, setBackendStatus] = useState(null);
 * useBackendStatus(setBackendStatus);
 */
export function useBackendStatus(setStatus) {
  React.useEffect(() => {
    // Initial check
    healthCheck().then(setStatus);

    // Poll every 30 seconds
    const interval = setInterval(() => {
      healthCheck().then(setStatus).catch(console.error);
    }, 30000);

    return () => clearInterval(interval);
  }, [setStatus]);
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  healthCheck,
  predictRisk,
  explainPrediction,
  whatIfAnalysis,
  batchPredict,
  formatErrorMessage,
  isBackendOffline,
  isModelUnavailable,
  getStatusDisplay,
  formatPrediction,
  useBackendStatus,
};