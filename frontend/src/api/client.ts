import { PatientInput, PredictionResponse, ExplainabilityResponse, HealthResponse } from '../types/api';

const DEFAULT_URL = 'https://cardiolens-ai-za8w.onrender.com';
const rawEnvUrl = (import.meta.env.VITE_API_URL || DEFAULT_URL)
  .replace(/[\[\]"']/g, '')
  .trim()
  .replace(/\/+$/, '');

const BASE_URL = rawEnvUrl.endsWith('/api') ? rawEnvUrl : `${rawEnvUrl}/api`;

export async function checkHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${BASE_URL}/health`, { method: 'GET' });
    if (!res.ok) throw new Error(`Health status code: ${res.status}`);
    return await res.json();
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
}

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