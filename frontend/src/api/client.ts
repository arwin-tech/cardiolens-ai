/**
 * CardioLens AI API client.
 *
 * Thin, typed wrappers around the FastAPI backend. Requests carry no
 * credentials (the API is stateless), and server errors, including FastAPI's
 * field-by-field validation errors, become readable messages.
 */
import {
  BatchResponse,
  CounterfactualResponse,
  ExplainabilityResponse,
  ModelId,
  ModelInfo,
  MonitoringSummary,
  PatientInput,
  PredictionResponse,
  WhatIfResponse,
} from '../types/api';

const API_BASE_URL: string = import.meta.env.VITE_API_URL || 'https://cardiolens-ai-za8w.onrender.com';

// /api/status first: some ad-blockers block URLs that contain "health"
const HEALTH_PATHS = ['/api/status', '/api/health', '/health'];

const OFFLINE_MESSAGE =
  "Couldn't reach the model server. It may still be waking up, so try again in a moment.";

export interface HealthResult {
  ok: boolean;
  status: string;
  modelLoaded: boolean;
  endpoint: string | null;
}

/** Checks each health URL in turn; resolves (never rejects) with the first answer. */
export async function healthCheck(timeoutMs = 15000): Promise<HealthResult> {
  for (const path of HEALTH_PATHS) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      // No Content-Type header on GET, so the browser doesn't send an extra CORS preflight
      const response = await fetch(`${API_BASE_URL}${path}`, {
        signal: controller.signal,
        credentials: 'omit',
        headers: { Accept: 'application/json' },
        cache: 'no-store',
      });
      if (response.ok) {
        const data = await response.json();
        return { ok: true, status: data.status ?? 'unknown', modelLoaded: Boolean(data.model_loaded), endpoint: path };
      }
    } catch {
      // network error or timeout: try the next URL
    } finally {
      clearTimeout(timer);
    }
  }
  return { ok: false, status: 'offline', modelLoaded: false, endpoint: null };
}

interface ValidationIssue {
  loc?: (string | number)[];
  msg?: string;
}

/** Builds a readable message from an error response. */
async function errorMessage(response: Response, fallback: string): Promise<string> {
  const body = await response.json().catch(() => null);
  const detail = body?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    // FastAPI validation errors look like [{ loc: ['body', 'modified', 'ap_hi'], msg: '...' }]
    return (detail as ValidationIssue[])
      .map((issue) => {
        const field = (issue.loc ?? []).filter((part) => part !== 'body').join(' ');
        const msg = (issue.msg ?? 'Invalid value').replace(/^Value error, /, '');
        return field ? `${field}: ${msg}` : msg;
      })
      .join('; ');
  }
  return `${fallback} (HTTP ${response.status})`;
}

async function send<T>(path: string, init: RequestInit, fallback: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { ...init, credentials: 'omit' });
  } catch {
    throw new Error(OFFLINE_MESSAGE);
  }
  if (!response.ok) throw new Error(await errorMessage(response, fallback));
  return (await response.json()) as T;
}

function postJson<T>(path: string, body: unknown, fallback: string): Promise<T> {
  return send<T>(
    path,
    { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' }, body: JSON.stringify(body) },
    fallback,
  );
}

const withModel = (path: string, model: ModelId) => `${path}?model=${encodeURIComponent(model)}`;

export function predictRisk(patient: PatientInput, model: ModelId = 'kaggle'): Promise<PredictionResponse> {
  return postJson(withModel('/api/predict', model), patient, 'Prediction failed');
}

export function explainPrediction(patient: PatientInput, model: ModelId = 'kaggle'): Promise<ExplainabilityResponse> {
  return postJson(withModel('/api/explain', model), patient, 'Explanation failed');
}

export function whatIfAnalysis(
  original: PatientInput,
  modified: PatientInput,
  model: ModelId = 'kaggle',
): Promise<WhatIfResponse> {
  return postJson(withModel('/api/what-if', model), { original, modified }, 'What-if analysis failed');
}

export function getCounterfactuals(patient: PatientInput, model: ModelId = 'kaggle'): Promise<CounterfactualResponse> {
  return postJson(withModel('/api/counterfactuals', model), patient, 'Could not load suggestions');
}

export function getModels(): Promise<{ default: ModelId; models: ModelInfo[] }> {
  return send('/api/models', { headers: { Accept: 'application/json' } }, 'Could not load models');
}

export function getMonitoring(days = 30): Promise<MonitoringSummary> {
  return send(`/api/monitoring?days=${days}`, { headers: { Accept: 'application/json' } }, 'Could not load monitoring');
}

export async function batchPredict(csvFile: File, model: ModelId = 'kaggle'): Promise<BatchResponse> {
  if (!csvFile.name.toLowerCase().endsWith('.csv')) {
    throw new Error('Please choose a CSV file.');
  }
  const formData = new FormData();
  formData.append('file', csvFile);
  // No Content-Type header: the browser sets the multipart boundary itself
  return send(withModel('/api/batch-predict', model), { method: 'POST', body: formData }, 'Batch scoring failed');
}

// Older name kept for existing imports
export { whatIfAnalysis as runWhatIf };
