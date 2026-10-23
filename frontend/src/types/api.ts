export interface PatientInput {
  age: number;
  gender: number;
  height: number;
  weight: number;
  ap_hi: number;
  ap_lo: number;
  cholesterol: number;
  gluc: number;
  smoke: number;
  alco: number;
  active: number;
  bp_meds?: number;
  lipid_meds?: number;
}

export type ModelId = 'kaggle' | 'nhanes';

export interface PredictionResponse {
  risk_probability: number;
  risk_percentage: number;
  risk_category: string;
  bmi: number;
  patient_features: PatientInput;
  model_version: string;
  disclaimer: string;
  model_id?: ModelId;
  model_name?: string;
  risk_range?: [number, number] | null;
}

export interface ModelInfo {
  id: ModelId;
  name: string;
  description: string;
  features: string[];
  default: boolean;
  uncertainty: { method: string; calibration_patients: number } | null;
}

export interface CounterfactualChange {
  feature: string;
  label: string;
}

export interface CounterfactualResponse {
  baseline_risk: number;
  plans: { changes: CounterfactualChange[]; risk: number; delta: number }[];
  note: string;
}

export interface LatencyStats {
  requests: number;
  histogram: { bucket: string; count: number | null }[];
  median: string | null;
  p95: string | null;
}

export interface DriftStats {
  status: 'ok' | 'not_enough_data';
  have: number;
  needed: number;
  features: Record<string, { psi: number; status: string }>;
}

export interface MonitoringSummary {
  window_days: number;
  storage: 'persistent' | 'in_memory';
  predictions_total: number;
  predictions_by_model: Record<string, number>;
  predictions_per_day: { day: string; predictions: number }[];
  requests_by_endpoint: Record<string, number>;
  risk_categories: Record<string, Record<string, number | null>>;
  response_times: Record<string, LatencyStats>;
  drift: Record<string, DriftStats>;
  privacy: { stored: string; smallest_count_shown: number; drift_needs_predictions: number };
}

export interface SHAPFeature {
  feature: string;
  value: number;
  shap_value: number;
  impact: 'positive' | 'negative';
  feature_rank: number;
}

export interface ExplainabilityResponse {
  features: SHAPFeature[];
  base_value: number;
  prediction_value?: number;
  disclaimer?: string;
}

export interface WhatIfResponse {
  original_risk: number;
  simulated_risk: number;
  risk_delta: number;
  original_ap_hi: number;
  target_ap_hi: number;
}

export interface BatchResponse {
  total_records: number;
  scored_records: number;
  skipped_records: number;
  high_risk_count: number;
  results: Array<Record<string, string | number | boolean | null>>;
  skipped: { line: number; reason: string }[];
}

export interface HealthResponse {
  status: string;
  service: string;
  model_loaded: boolean;
  version: string;
}
