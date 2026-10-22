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
}

export interface PredictionResponse {
  risk_probability: number;
  risk_percentage: number;
  risk_category: string;
  bmi: number;
  patient_features: PatientInput;
  model_version: string;
  disclaimer: string;
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
  high_risk_count: number;
  results: Array<Record<string, string | number | boolean | null>>;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  timestamp: string;
}
