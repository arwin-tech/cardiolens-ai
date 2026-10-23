import { explainPrediction, predictRisk } from '../api/client';
import { ExplainabilityResponse, ModelId, PatientInput, PredictionResponse } from '../types/api';

export interface Assessment {
  prediction: PredictionResponse;
  explanation: ExplainabilityResponse;
}

/** Score a patient, explain the score, and keep the result for the Results page. */
export async function runAssessment(data: PatientInput, model: ModelId = 'kaggle'): Promise<Assessment> {
  const prediction: PredictionResponse = await predictRisk(data, model);
  const explanation: ExplainabilityResponse = await explainPrediction(data, model);
  sessionStorage.setItem('lastPrediction', JSON.stringify(prediction));
  sessionStorage.setItem('lastExplanation', JSON.stringify(explanation));
  return { prediction, explanation };
}

/** The last result in this tab, or null if there isn't a usable one. */
export function readCachedAssessment(): Assessment | null {
  try {
    const p = sessionStorage.getItem('lastPrediction');
    const e = sessionStorage.getItem('lastExplanation');
    if (!p || !e) return null;
    const prediction: PredictionResponse = JSON.parse(p);
    const explanation: ExplainabilityResponse = JSON.parse(e);
    if (!prediction?.patient_features || !Array.isArray(explanation?.features)) return null;
    return { prediction, explanation };
  } catch {
    return null;
  }
}
