import { PatientInput, PredictionResponse, ExplainabilityResponse, HealthResponse } from '../types/api';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/api/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export async function predictRisk(patient: PatientInput): Promise<PredictionResponse> {
  const res = await fetch(`${API_URL}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patient),
  });
  if (!res.ok) throw new Error('Prediction API call failed');
  return res.json();
}

export async function explainPrediction(patient: PatientInput): Promise<ExplainabilityResponse> {
  const res = await fetch(`${API_URL}/api/explain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patient),
  });
  if (!res.ok) throw new Error('Explainability API call failed');
  return res.json();
}

export async function runWhatIf(original: PatientInput, modified: PatientInput) {
  const response = await fetch(`${API_URL}/api/what-if`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      original: original,
      modified: modified,
    }),
  });

  if (!response.ok) {
    const errorDetail = await response.json();
    console.error('FastAPI Validation Detail:', errorDetail);
    throw new Error('What-If API call failed');
  }

  return response.json();
}