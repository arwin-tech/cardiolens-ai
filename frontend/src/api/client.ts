import { PatientInput, PredictionResponse, ExplainabilityResponse, HealthResponse } from '../types/api';

// Ensures BASE_URL points directly to the /api namespace without trailing slashes
const RAW_URL = (import.meta.env.VITE_API_URL || 'https://cardiolens-ai-za8w.onrender.com').replace(/\/+$/, '');
const BASE_URL = RAW_URL.endsWith('/api') ? RAW_URL : `${RAW_URL}/api`;

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export async function predictRisk(patient: PatientInput): Promise<PredictionResponse> {
  const res = await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patient),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Prediction API call failed');
  }
  return res.json();
}

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