import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Download, FileText, Upload, CheckCircle } from 'lucide-react';
import RiskGauge from '../components/RiskGauge';
import SHAPChart from '../components/SHAPChart';
import WhatIfComparison from '../components/WhatIfComparison';
import { runWhatIf } from '../api/client';
import { PredictionResponse, ExplainabilityResponse, WhatIfResponse, PatientInput } from '../types/api';

// -----------------------------------------------------------------------------
// 1. SUB-COMPONENT: CLINICAL REPORT & BATCH SCREENING TABS
// -----------------------------------------------------------------------------
interface ClinicalReportProps {
  patientData?: PatientInput;
  riskScore: number;
}

function ClinicalReportAndBatch({ patientData, riskScore }: ClinicalReportProps) {
  const [activeTab, setActiveTab] = useState<'report' | 'batch'>('report');
  const [batchFile, setBatchFile] = useState<File | null>(null);

  const reportContent = `
================================================================================
                    CARDIOVASCULAR RISK EVALUATION REPORT
================================================================================

PATIENT METRICS & DEMOGRAPHICS:
--------------------------------------------------------------------------------
* Age:                    ${patientData?.age ? Math.round(patientData.age / 365) : 54} years
* Height / Weight:        ${patientData?.height || 165} cm / ${patientData?.weight || 72} kg
* Body Mass Index (BMI):  ${patientData?.weight && patientData?.height ? (patientData.weight / Math.pow(patientData.height / 100, 2)).toFixed(2) : '26.45'} kg/m²

CLINICAL BIOMARKERS:
--------------------------------------------------------------------------------
* Systolic BP:            ${patientData?.ap_hi || 130} mmHg
* Diastolic BP:           ${patientData?.ap_lo || 85} mmHg
* Cholesterol Level:      ${patientData?.cholesterol || 1}
* Glucose Level:          ${patientData?.gluc || 1}

RISK ASSESSMENT RESULTS:
--------------------------------------------------------------------------------
* Estimated Disease Risk: ${(riskScore * 100).toFixed(2)}%
* Risk Stratification:   ${riskScore >= 0.5 ? "HIGH RISK" : "MODERATE / LOW RISK"}

RECOMMENDED CLINICAL ACTIONS:
--------------------------------------------------------------------------------
${riskScore >= 0.5 ? "1. Lifestyle modification: Weight management, sodium restriction.\n2. Antihypertensive/statin therapy consideration.\n3. Follow-up diagnostic workup (ECG/Echocardiogram)." : "1. Routine follow-up visits.\n2. Encourage maintained daily physical activity.\n3. Annual lipid profiling."}
================================================================================
`;

  const handleDownload = () => {
    const element = document.createElement("a");
    const file = new Blob([reportContent], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = "cardio_risk_report_patient.txt";
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <div className="mt-8 p-6 bg-white border border-[#E8DFD9] rounded-xl shadow-sm space-y-6">
      <div className="flex space-x-4 border-b border-gray-200 pb-3">
        <button
          onClick={() => setActiveTab('report')}
          className={`flex items-center space-x-2 font-bold px-4 py-2 rounded-lg transition ${activeTab === 'report' ? 'bg-[#8B2D2D] text-white' : 'text-gray-600 hover:bg-gray-100'}`}
        >
          <FileText className="w-4 h-4" />
          <span>Clinical EHR Report Generator</span>
        </button>

        <button
          onClick={() => setActiveTab('batch')}
          className={`flex items-center space-x-2 font-bold px-4 py-2 rounded-lg transition ${activeTab === 'batch' ? 'bg-[#8B2D2D] text-white' : 'text-gray-600 hover:bg-gray-100'}`}
        >
          <Upload className="w-4 h-4" />
          <span>Batch Patient Profiling</span>
        </button>
      </div>

      {activeTab === 'report' && (
        <div className="space-y-4">
          <p className="text-sm text-gray-600">
            Export formatted summary report ready for Electronic Health Records (EHR).
          </p>

          <pre className="p-4 bg-[#F9F7F4] border border-[#E8DFD9] rounded-lg text-xs font-mono overflow-x-auto text-gray-800">
            {reportContent}
          </pre>

          <button
            onClick={handleDownload}
            className="flex items-center space-x-2 px-5 py-2.5 bg-[#8B2D2D] hover:bg-[#722323] text-white font-semibold rounded-lg shadow-sm transition"
          >
            <Download className="w-4 h-4" />
            <span>Download Clinical Report (.txt)</span>
          </button>
        </div>
      )}

      {activeTab === 'batch' && (
        <div className="space-y-4">
          <p className="text-sm text-gray-600">
            Upload a CSV file containing patient records to process population-level risk.
          </p>

          <input
            type="file"
            accept=".csv"
            onChange={(e) => setBatchFile(e.target.files?.[0] || null)}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-[#8B2D2D] file:text-white hover:file:bg-[#722323]"
          />

          {batchFile && (
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg flex items-center space-x-2 text-green-800 text-sm font-semibold">
              <CheckCircle className="w-5 h-5 text-green-600" />
              <span>
                Loaded dataset: {batchFile.name} ready for batch processing!
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// -----------------------------------------------------------------------------
// 2. MAIN PAGE COMPONENT
// -----------------------------------------------------------------------------
export default function Results() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplainabilityResponse | null>(null);
  const [whatIfData, setWhatIfData] = useState<WhatIfResponse | null>(null);
  const [simulatedSysBP, setSimulatedSysBP] = useState<number>(120);
  const navigate = useNavigate();

  useEffect(() => {
    const pCache = sessionStorage.getItem('lastPrediction');
    const eCache = sessionStorage.getItem('lastExplanation');

    if (!pCache || !eCache) {
      navigate('/analyze');
      return;
    }

    const parsedPred: PredictionResponse = JSON.parse(pCache);

    setPrediction(parsedPred);
    setExplanation(JSON.parse(eCache));

    setSimulatedSysBP(
      parsedPred.patient_features.ap_hi > 120
        ? 120
        : parsedPred.patient_features.ap_hi - 10
    );
  }, [navigate]);

  if (!prediction || !explanation) return null;

  const handleSimulate = async () => {
  if (!prediction) return;

  const currentFeatures = prediction.patient_features;

  const modifiedFeatures = {
    ...currentFeatures,
    ap_hi: simulatedSysBP,
    ap_lo: Math.min(currentFeatures.ap_lo, 80),
  };

  try {
    const res = await runWhatIf(currentFeatures, modifiedFeatures);
    setWhatIfData(res);
  } catch (e) {
    console.error("Simulation failed:", e);
  }
};
  

  return (
    <div className="max-w-5xl mx-auto px-6 py-10 space-y-8">

      <div className="flex justify-between items-center border-b border-[#E8DFD9] pb-4">
        <div>
          <h2 className="text-3xl font-extrabold text-[#8B2D2D]">
            Patient Assessment Dashboard
          </h2>

          <p className="text-xs text-gray-500 mt-1">
            Model Version: {prediction.model_version}
          </p>
        </div>

        <button
          onClick={() => navigate('/analyze')}
          className="px-4 py-2 bg-[#FFFBF8] border border-[#E8DFD9] text-sm text-gray-700 font-semibold rounded-lg hover:text-[#8B2D2D] transition"
        >
          New Assessment
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

        <RiskGauge
          percentage={prediction.risk_percentage}
          category={prediction.risk_category}
        />

        <div className="md:col-span-2 p-6 bg-[#FFFBF8] border border-[#E8DFD9] rounded-medical shadow-sm space-y-4">

          <h4 className="text-lg font-bold text-[#8B2D2D]">
            Calculated Clinical Metrics
          </h4>

          <div className="grid grid-cols-2 gap-4 text-sm">

            <div className="p-3 bg-white border border-[#E8DFD9] rounded-lg">
              <span className="text-xs text-gray-400 font-semibold block uppercase">
                Body Mass Index (BMI)
              </span>

              <span className="text-lg font-bold text-gray-800">
                {prediction.bmi.toFixed(1)} kg/m²
              </span>
            </div>

            <div className="p-3 bg-white border border-[#E8DFD9] rounded-lg">
              <span className="text-xs text-gray-400 font-semibold block uppercase">
                Blood Pressure
              </span>

              <span className="text-lg font-bold text-gray-800">
                {prediction.patient_features.ap_hi} / {prediction.patient_features.ap_lo} mmHg
              </span>
            </div>

          </div>

          <p className="text-xs text-gray-500 bg-amber-500/10 border border-amber-500/30 p-3 rounded-md leading-relaxed">
            {prediction.disclaimer}
          </p>

        </div>
      </div>

      <SHAPChart features={explanation.features} />

      <div className="p-6 bg-[#FFFBF8] border border-[#E8DFD9] rounded-medical shadow-sm space-y-6">

        <h4 className="text-lg font-bold text-[#8B2D2D]">
          Interactive Counterfactual Simulation
        </h4>

        <div className="flex items-center space-x-4">

          <label className="text-xs font-semibold text-gray-600">
            Simulate Target Systolic BP (mmHg):
          </label>

          <input
            type="number"
            value={simulatedSysBP}
            onChange={(e) => setSimulatedSysBP(Number(e.target.value))}
            className="p-2 border border-[#E8DFD9] rounded-md text-sm w-32"
          />

          <button
            onClick={handleSimulate}
            className="px-4 py-2 bg-[#8B2D2D] hover:bg-[#722323] text-white text-sm font-semibold rounded-md transition"
          >
            Execute What-If Simulation
          </button>

        </div>

        {whatIfData && (
          <WhatIfComparison data={whatIfData} />
        )}

      </div>

      {/* Integrated Streamlit Features: EHR Report & Batch Upload */}
      <ClinicalReportAndBatch
        patientData={prediction.patient_features}
        riskScore={prediction.risk_percentage / 100}
      />

    </div>
  );
}