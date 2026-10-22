import { PatientInput } from '../types/api';

// Example patients across the risk range (model output: about 13%, 47% and 91%)
export const EXAMPLES: { label: string; color: string; data: PatientInput }[] = [
  { label: 'Low-risk example', color: '#6FAE7E', data: { age: 34, gender: 1, height: 168, weight: 60, ap_hi: 115, ap_lo: 75, cholesterol: 1, gluc: 1, smoke: 0, alco: 0, active: 1 } },
  { label: 'Moderate example', color: '#D0A15A', data: { age: 53, gender: 2, height: 176, weight: 84, ap_hi: 135, ap_lo: 85, cholesterol: 1, gluc: 1, smoke: 1, alco: 0, active: 1 } },
  { label: 'High-risk example', color: '#C8322B', data: { age: 61, gender: 2, height: 172, weight: 98, ap_hi: 165, ap_lo: 100, cholesterol: 3, gluc: 2, smoke: 1, alco: 0, active: 0 } },
];
