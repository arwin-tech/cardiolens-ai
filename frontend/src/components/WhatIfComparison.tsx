import { WhatIfResponse } from '../types/api';

export default function WhatIfComparison({
  data,
  diastolic,
}: {
  data: WhatIfResponse | null;
  diastolic?: { from: number; to: number };
}) {
  if (!data) return null;

  const { original_risk: origRisk, simulated_risk: simRisk } = data;
  const delta = data.risk_delta ?? simRisk - origRisk;
  const reduced = delta <= 0;
  const deltaColor = reduced ? '#6FAE7E' : '#D9574B';

  return (
    <div className="mt-6 space-y-4" aria-live="polite">
      <div className="grid gap-4 sm:grid-cols-3">
        <div className="stat">
          <span className="stat__label">Current risk</span>
          <span className="stat__value">{origRisk.toFixed(1)}%</span>
        </div>
        <div className="stat">
          <span className="stat__label">Simulated risk</span>
          <span className="stat__value">{simRisk.toFixed(1)}%</span>
        </div>
        <div className="stat" style={{ borderColor: `${deltaColor}88` }}>
          <span className="stat__label">Change</span>
          <span className="stat__value" style={{ color: deltaColor }}>
            {delta > 0 ? '+' : delta < 0 ? '−' : ''}
            {Math.abs(delta).toFixed(1)} points
          </span>
        </div>
      </div>
      <p className="text-stone">
        Blood pressure changed from{' '}
        <strong className="font-medium text-parchment">
          {data.original_ap_hi}
          {diastolic ? `/${diastolic.from}` : ''} mmHg
        </strong>{' '}
        to{' '}
        <strong className="font-medium text-parchment">
          {data.target_ap_hi}
          {diastolic ? `/${diastolic.to}` : ''} mmHg
        </strong>
        .
      </p>
    </div>
  );
}
