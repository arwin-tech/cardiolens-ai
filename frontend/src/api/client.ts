import { PatientInput, PredictionResponse, ExplainabilityResponse, HealthResponse } from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || "https://cardiolens-ai-za8w.onrender.com";

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

export async function healthCheck(): Promise<HealthResponse> {
  for (const url of HEALTH_ENDPOINTS) {
    try {
      const response = await fetch(url, {
        method: "GET",
        headers: { "Accept": "application/json", "Content-Type": "application/json" },
        cache: "no-store",
      });
      if (response.ok) {
        return await response.json();
      }
    } catch (err) {
      continue;
    }
  }
  throw new Error('All health endpoints failed');
}

export async function checkHealth(): Promise<HealthResponse> {
  return await healthCheck();
}

export const predictRisk = async (data: PatientInput): Promise<PredictionResponse> => {
  const response = await fetch(PREDICTION_ENDPOINTS.predict, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || "Prediction API call failed");
  }
  return response.json();
};

export async function explainPrediction(patient: PatientInput): Promise<ExplainabilityResponse> {
  const response = await fetch(PREDICTION_ENDPOINTS.explain, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patient),
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || "Explainability API call failed");
  }
  return response.json();
}

export async function runWhatIf(original: PatientInput, modified: PatientInput) {
  const response = await fetch(PREDICTION_ENDPOINTS.whatIf, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ original, modified }),
  });
  if (!response.ok) {
    const errorDetail = await response.json().catch(() => ({}));
    throw new Error(errorDetail.detail || "What-If API call failed");
  }
  return response.json();
}

export async function whatIfAnalysis(original: PatientInput, modified: PatientInput) {
  return await runWhatIf(original, modified);
}

export async function batchPredict(csvFile: File) {
  const formData = new FormData();
  formData.append("file", csvFile);
  const response = await fetch(PREDICTION_ENDPOINTS.batchPredict, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || "Batch prediction failed");
  }
  return response.json();
}

export default {
  healthCheck,
  checkHealth,
  predictRisk,
  explainPrediction,
  runWhatIf,
  whatIfAnalysis,
  batchPredict,
};