import { PatientInput, PredictionResponse, ExplainabilityResponse, HealthResponse } from '../types/api';

// Normalize BASE_URL and ensure no trailing slash
const BASE_URL = (import.meta.env.VITE_API_URL || 'https://cardiolens-ai-za8w.onrender.com').replace(/\/+$/, '');

/**
 * Health check ping to verify API status & handle cold-starts
 */
export async function checkHealth(): Promise<HealthResponse> {
  try {
    // Tries /api/health first, falls back to /health if needed
    const res = await fetch(`${BASE_URL}/api/health`, { method: 'GET' })
      .catch(() => fetch(`${BASE_URL}/health`, { method: 'GET' }));

    if (!res.ok) throw new Error(`Health status code: ${res.status}`);
    return await res.json();
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
}

/**
 * Submit patient metrics for cardiovascular risk scoring
 */
export const predictRisk = async (data: PatientInput): Promise<PredictionResponse> => {
  const response = await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Prediction API call failed');
  }
  return response.json();
};

/**
 * Fetch SHAP feature importance explainability breakdown
 */
export async function explainPrediction(patient: PatientInput): Promise<ExplainabilityResponse> {
  const res = await fetch(`${BASE_URL}/explain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patient),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Explainability API call failed');
  }
  return res.json();
}

/**
 * Execute dynamic what-if simulation comparing original vs altered metrics
 */
export async function runWhatIf(original: PatientInput, modified: PatientInput) {
  const response = await fetch(`${BASE_URL}/what-if`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ original, modified }),
  });

  if (!response.ok) {
    const errorDetail = await response.json().catch(() => ({}));
    console.error('FastAPI Validation Detail:', errorDetail);
    throw new Error(errorDetail.detail || 'What-If API call failed');
  }

  return response.json();
}