import React from 'react';

export default function WhatIfComparison({ data }: { data: any }) {
  if (!data) return null;

  const origRisk = data.original_risk ?? data.original_risk_percentage ?? 0;
  const simRisk = data.simulated_risk ?? data.simulated_risk_percentage ?? 0;
  const delta = data.risk_delta ?? (simRisk - origRisk);
  const origBp = data.original_ap_hi ?? data.original_features?.ap_hi ?? '--';
  const targetBp = data.target_ap_hi ?? data.modified_features?.ap_hi ?? '--';

  const isReduced = delta <= 0;

  return (
    <div className="mt-6 p-5 bg-white border border-[#E8DFD9] rounded-xl shadow-sm space-y-4">
      <h5 className="text-md font-bold text-[#8B2D2D]">Counterfactual Simulation Results</h5>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
        <div className="p-4 bg-[#F9F7F4] rounded-lg border border-[#E8DFD9]">
          <span className="text-xs text-gray-500 uppercase font-semibold block">Baseline Risk</span>
          <span className="text-xl font-extrabold text-gray-800">{origRisk.toFixed(1)}%</span>
        </div>

        <div className="p-4 bg-[#F9F7F4] rounded-lg border border-[#E8DFD9]">
          <span className="text-xs text-gray-500 uppercase font-semibold block">Simulated Risk</span>
          <span className="text-xl font-extrabold text-[#8B2D2D]">{simRisk.toFixed(1)}%</span>
        </div>

        <div className={`p-4 rounded-lg border ${isReduced ? 'bg-green-50 border-green-200 text-green-800' : 'bg-red-50 border-red-200 text-red-800'}`}>
          <span className="text-xs uppercase font-semibold block">Net Risk Delta</span>
          <span className="text-xl font-extrabold">
            {delta > 0 ? `+${delta.toFixed(1)}%` : `${delta.toFixed(1)}%`}
          </span>
        </div>
      </div>

      <p className="text-xs text-gray-600 bg-[#F9F7F4] p-3 rounded-lg border border-[#E8DFD9]">
        Targeted intervention: Modified Systolic Blood Pressure from <strong>{origBp} mmHg</strong> to <strong>{targetBp} mmHg</strong>.
      </p>
    </div>
  );
}