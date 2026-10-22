// Plain-language names and value formats for the model's 11 input features.
export const FEATURE_LABELS: Record<string, string> = {
  age: 'Age',
  gender: 'Gender',
  height: 'Height',
  weight: 'Weight',
  ap_hi: 'Systolic BP',
  ap_lo: 'Diastolic BP',
  cholesterol: 'Cholesterol',
  gluc: 'Glucose',
  smoke: 'Smoking',
  alco: 'Alcohol',
  active: 'Physical activity',
};

const LEVELS = ['', 'Normal', 'Above normal', 'Well above normal'];

export function formatFeatureValue(feature: string, value: number): string {
  switch (feature) {
    case 'age':
      return `${Math.round(value)} years`;
    case 'gender':
      return value === 2 ? 'Male' : 'Female';
    case 'height':
      return `${Math.round(value)} cm`;
    case 'weight':
      return `${Number(value.toFixed(1))} kg`;
    case 'ap_hi':
    case 'ap_lo':
      return `${Math.round(value)} mmHg`;
    case 'cholesterol':
    case 'gluc':
      return LEVELS[Math.round(value)] ?? String(value);
    case 'smoke':
    case 'alco':
      return value ? 'Yes' : 'No';
    case 'active':
      return value ? 'Active' : 'Sedentary';
    default:
      return String(value);
  }
}

// Risk levels returned by the API, mapped to the palette.
export const RISK_COLORS: Record<string, string> = {
  low: '#6FAE7E',
  moderate: '#D0A15A',
  elevated: '#C9643F',
  high: '#C8322B',
};

export const riskColor = (category: string) => RISK_COLORS[category.toLowerCase()] ?? '#C8322B';
