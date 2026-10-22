import { useState } from 'react';
import { Download, FileText, Upload } from 'lucide-react';
import RiskGauge, { EmptyDial } from '../components/RiskGauge';
import SHAPChart from '../components/SHAPChart';
import WhatIfComparison from '../components/WhatIfComparison';
import { PageBackdrop, PageHeader } from '../components/Backdrop';
import { FramedPanel, OrnateButton, OrnateLink } from '../components/ornaments';
import { batchPredict, runWhatIf } from '../api/client';
import { useBackendState } from '../context/BackendStatus';
import { formatFeatureValue } from '../lib/labels';
import { EXAMPLES } from '../lib/examples';
import { Assessment, readCachedAssessment, runAssessment } from '../lib/assess';
import { BatchResponse, PatientInput, PredictionResponse, WhatIfResponse } from '../types/api';

const BATCH_COLUMNS = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'];

const SAMPLE_CSV = [
  BATCH_COLUMNS.join(','),
  '34,1,168,60,115,75,1,1,0,0,1',
  '53,2,176,84,135,85,1,1,1,0,1',
  '61,2,172,98,165,100,3,2,1,0,0',
  '45,1,160,72,128,82,2,1,0,0,1',
  '67,2,170,88,150,95,2,1,0,1,0',
].join('\n');

function downloadText(content: string, filename: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function toCsv(rows: BatchResponse['results']) {
  if (!rows.length) return '';
  const cols = Object.keys(rows[0]);
  const cell = (v: unknown) => {
    const s = v === null || v === undefined ? '' : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [cols.join(','), ...rows.map((r) => cols.map((c) => cell(r[c])).join(','))].join('\n');
}

// -----------------------------------------------------------------------------
// Report download and batch scoring
// -----------------------------------------------------------------------------
interface ReportProps {
  patient: PatientInput;
  riskPercentage: number;
  category: string;
  bmi: number;
}

function ReportAndBatch({ patient, riskPercentage, category, bmi }: ReportProps) {
  const [activeTab, setActiveTab] = useState<'report' | 'batch'>('report');
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchError, setBatchError] = useState<string | null>(null);
  const [batchResult, setBatchResult] = useState<BatchResponse | null>(null);
  const backend = useBackendState();

  const line = (label: string, value: string) => `* ${(label + ':').padEnd(26)}${value}`;
  const reportContent = [
    '================================================================================',
    '                    CARDIOVASCULAR RISK EVALUATION REPORT',
    '================================================================================',
    '',
    'PATIENT DETAILS',
    '--------------------------------------------------------------------------------',
    line('Age', formatFeatureValue('age', patient.age)),
    line('Gender', formatFeatureValue('gender', patient.gender)),
    line('Height / weight', `${patient.height} cm / ${patient.weight} kg`),
    line('Body mass index (BMI)', `${bmi.toFixed(1)} kg/m²`),
    '',
    'CLINICAL MEASUREMENTS',
    '--------------------------------------------------------------------------------',
    line('Blood pressure', `${patient.ap_hi} / ${patient.ap_lo} mmHg`),
    line('Cholesterol', formatFeatureValue('cholesterol', patient.cholesterol)),
    line('Glucose', formatFeatureValue('gluc', patient.gluc)),
    line('Smoker', formatFeatureValue('smoke', patient.smoke)),
    line('Alcohol intake', formatFeatureValue('alco', patient.alco)),
    line('Physical activity', formatFeatureValue('active', patient.active)),
    '',
    'MODEL RESULT',
    '--------------------------------------------------------------------------------',
    line('Predicted risk', `${riskPercentage.toFixed(1)}%`),
    line('Risk level', category),
    '',
    'Research-grade prediction from a statistical model. Not a medical diagnosis.',
    '================================================================================',
  ].join('\n');

  const handleScore = async () => {
    if (!batchFile) return;
    setBatchLoading(true);
    setBatchError(null);
    setBatchResult(null);
    try {
      const result: BatchResponse = await batchPredict(batchFile);
      setBatchResult(result);
    } catch (err: any) {
      setBatchError(err?.message || 'Batch scoring failed. Check that the file has the required columns.');
    } finally {
      setBatchLoading(false);
    }
  };

  const preview = batchResult?.results.slice(0, 10) ?? [];

  return (
    <FramedPanel corners={false} aria-label="Report and batch scoring">
      <div className="tabs" role="tablist">
        <button
          type="button"
          role="tab"
          className="tab"
          aria-selected={activeTab === 'report'}
          onClick={() => setActiveTab('report')}
        >
          <FileText className="h-4 w-4" aria-hidden="true" /> Patient report
        </button>
        <button
          type="button"
          role="tab"
          className="tab"
          aria-selected={activeTab === 'batch'}
          onClick={() => setActiveTab('batch')}
        >
          <Upload className="h-4 w-4" aria-hidden="true" /> Batch scoring
        </button>
      </div>

      {activeTab === 'report' && (
        <div className="mt-6 space-y-5" role="tabpanel">
          <p className="text-stone">A plain-text summary of this assessment, ready to save or attach to a record.</p>
          <pre className="report">{reportContent}</pre>
          <OrnateButton
            type="button"
            tone="umber"
            small
            onClick={() => downloadText(reportContent, 'cardiolens_report.txt', 'text/plain')}
          >
            <Download className="h-4 w-4" aria-hidden="true" /> Download report (.txt)
          </OrnateButton>
        </div>
      )}

      {activeTab === 'batch' && (
        <div className="mt-6 space-y-5" role="tabpanel">
          <p className="text-stone">
            Score many patients at once from a CSV file. It needs these columns, with age in years:{' '}
            <span className="text-parchment">{BATCH_COLUMNS.join(', ')}</span>.{' '}
            <button
              type="button"
              className="text-gold-bright underline decoration-gold/50 underline-offset-4 hover:text-cream"
              onClick={() => downloadText(SAMPLE_CSV, 'cardiolens_sample.csv', 'text/csv')}
            >
              Download a sample file
            </button>
          </p>

          <div className="flex flex-wrap items-center gap-4">
            <input
              type="file"
              accept=".csv"
              aria-label="Choose a CSV file"
              className="file-input text-stone"
              onChange={(e) => {
                setBatchFile(e.target.files?.[0] || null);
                setBatchResult(null);
                setBatchError(null);
              }}
            />
            <OrnateButton type="button" tone="crimson" small disabled={!batchFile || batchLoading} onClick={handleScore}>
              {batchLoading ? (backend === 'online' ? 'Scoring…' : 'Waking up the model…') : 'Score this file'}
            </OrnateButton>
          </div>

          {batchError && (
            <p className="notice notice--error" role="alert">
              {batchError}
            </p>
          )}

          {batchResult && (
            <div className="space-y-4">
              <p className="text-parchment">
                Scored {batchResult.scored_records} of {batchResult.total_records} patients.{' '}
                {batchResult.high_risk_count} {batchResult.high_risk_count === 1 ? 'has' : 'have'} a predicted risk of 50% or higher.
              </p>
              {batchResult.skipped_records > 0 && (
                <div className="notice">
                  <p>
                    {batchResult.skipped_records} {batchResult.skipped_records === 1 ? 'row was' : 'rows were'} skipped
                    because they failed the same checks as the form:
                  </p>
                  <ul className="mt-2 list-disc space-y-0.5 pl-5 text-[0.95rem] text-stone">
                    {batchResult.skipped.slice(0, 5).map((s) => (
                      <li key={s.line}>
                        Line {s.line}: {s.reason}
                      </li>
                    ))}
                  </ul>
                  {batchResult.skipped_records > 5 && (
                    <p className="mt-1 text-[0.95rem] text-stone-2">and {batchResult.skipped_records - 5} more.</p>
                  )}
                </div>
              )}
              <div className="overflow-x-auto">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Age</th>
                      <th>Gender</th>
                      <th>Blood pressure</th>
                      <th>Cholesterol</th>
                      <th>Predicted risk</th>
                    </tr>
                  </thead>
                  <tbody>
                    {preview.map((row, i) => {
                      const risk = Number(row.risk_score) * 100;
                      return (
                        <tr key={i}>
                          <td>{i + 1}</td>
                          <td>{formatFeatureValue('age', Number(row.age))}</td>
                          <td>{formatFeatureValue('gender', Number(row.gender))}</td>
                          <td>
                            {row.ap_hi} / {row.ap_lo} mmHg
                          </td>
                          <td>{formatFeatureValue('cholesterol', Number(row.cholesterol))}</td>
                          <td className={risk >= 50 ? 'font-medium text-[#D9574B]' : 'text-parchment'}>{risk.toFixed(1)}%</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {batchResult.scored_records > preview.length && (
                <p className="text-[0.95rem] text-stone-2">Showing the first {preview.length} rows.</p>
              )}
              <OrnateButton
                type="button"
                tone="umber"
                small
                onClick={() => downloadText(toCsv(batchResult.results), 'cardiolens_scored.csv', 'text/csv')}
              >
                <Download className="h-4 w-4" aria-hidden="true" /> Download all results (.csv)
              </OrnateButton>
            </div>
          )}
        </div>
      )}
    </FramedPanel>
  );
}

// -----------------------------------------------------------------------------
// Empty state: nothing scored in this tab yet
// -----------------------------------------------------------------------------
function EmptyResults({ onResult }: { onResult: (a: Assessment) => void }) {
  const backend = useBackendState();
  const [running, setRunning] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runExample = async (label: string, data: PatientInput) => {
    setRunning(label);
    setError(null);
    try {
      onResult(await runAssessment(data));
    } catch (e: any) {
      setError(e?.message || 'Could not reach the model. Try again in a moment.');
      setRunning(null);
    }
  };

  return (
    <div className="mx-auto max-w-3xl px-5">
      <FramedPanel className="text-center" aria-labelledby="empty-title">
        <EmptyDial />
        <h2 id="empty-title" className="panel__title mt-6">
          No assessment yet
        </h2>
        <p className="mx-auto mt-3 max-w-md text-stone">
          Run an assessment and the risk score, the factors behind it and the what-if simulator will appear here.
        </p>
        <div className="mt-7 flex justify-center">
          <OrnateLink to="/analyze" tone="crimson">
            Run an assessment
          </OrnateLink>
        </div>

        <div className="mt-8 flex flex-wrap items-center justify-center gap-2.5">
          <span className="mr-1 text-stone">Or see an example result:</span>
          {EXAMPLES.map((ex) => (
            <button
              key={ex.label}
              type="button"
              className="chip"
              disabled={running !== null}
              onClick={() => runExample(ex.label, ex.data)}
            >
              <span className="chip__dot" style={{ background: ex.color }} aria-hidden="true" />
              {running === ex.label ? 'Scoring…' : ex.label}
            </button>
          ))}
        </div>

        {running && backend !== 'online' && (
          <p className="mt-5 text-[0.95rem] text-stone">
            The free server is waking up, which can take up to a minute. Your result will appear here.
          </p>
        )}
        {error && (
          <p className="notice notice--error mt-6" role="alert">
            {error}
          </p>
        )}
      </FramedPanel>
    </div>
  );
}

// -----------------------------------------------------------------------------
// Page
// -----------------------------------------------------------------------------
const defaultTargetBP = (p: PredictionResponse) => (p.patient_features.ap_hi > 120 ? 120 : p.patient_features.ap_hi - 10);

export default function Results() {
  const [result, setResult] = useState<Assessment | null>(readCachedAssessment);
  const [whatIfData, setWhatIfData] = useState<WhatIfResponse | null>(null);
  const [whatIfDiastolic, setWhatIfDiastolic] = useState<{ from: number; to: number } | undefined>();
  const [simulatedSysBP, setSimulatedSysBP] = useState<number>(() => (result ? defaultTargetBP(result.prediction) : 120));
  const [simLoading, setSimLoading] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);

  const showResult = (a: Assessment) => {
    setResult(a);
    setWhatIfData(null);
    setSimError(null);
    setSimulatedSysBP(defaultTargetBP(a.prediction));
  };

  if (!result) {
    return (
      <div className="page">
        <PageBackdrop />
        <PageHeader title="Patient assessment" />
        <EmptyResults onResult={showResult} />
      </div>
    );
  }

  const { prediction, explanation } = result;
  const patient = prediction.patient_features;

  const handleSimulate = async () => {
    const modifiedFeatures = {
      ...patient,
      ap_hi: simulatedSysBP,
      // keep diastolic below the new systolic so the change stays physiologically valid
      ap_lo: Math.min(patient.ap_lo, 80, simulatedSysBP - 1),
    };

    setSimLoading(true);
    setSimError(null);
    try {
      const res = await runWhatIf(patient, modifiedFeatures);
      setWhatIfData(res);
      setWhatIfDiastolic({ from: patient.ap_lo, to: modifiedFeatures.ap_lo });
    } catch (e: any) {
      setSimError(e?.message || 'The simulation failed. Try again in a moment.');
    } finally {
      setSimLoading(false);
    }
  };

  return (
    <div className="page">
      <PageBackdrop />
      <PageHeader
        title="Patient assessment"
        intro={`Scored by ${prediction.model_version}.`}
        action={
          <OrnateLink to="/analyze" tone="umber" small>
            New assessment
          </OrnateLink>
        }
      />

      <div className="mx-auto max-w-5xl space-y-7 px-5">
        <div className="grid gap-7 md:grid-cols-5">
          <FramedPanel className="md:col-span-2" aria-label="Risk score">
            <RiskGauge percentage={prediction.risk_percentage} category={prediction.risk_category} />
          </FramedPanel>

          <FramedPanel corners={false} className="md:col-span-3" aria-labelledby="measurements-title">
            <h2 id="measurements-title" className="panel__title">
              Key measurements
            </h2>
            <div className="rule" />
            <div className="grid grid-cols-2 gap-4">
              <div className="stat">
                <span className="stat__label">Body mass index</span>
                <span className="stat__value">
                  {prediction.bmi.toFixed(1)} <span className="text-base font-normal text-stone">kg/m²</span>
                </span>
              </div>
              <div className="stat">
                <span className="stat__label">Blood pressure</span>
                <span className="stat__value">
                  {patient.ap_hi}/{patient.ap_lo} <span className="text-base font-normal text-stone">mmHg</span>
                </span>
              </div>
              <div className="stat">
                <span className="stat__label">Age</span>
                <span className="stat__value">
                  {Math.round(patient.age)} <span className="text-base font-normal text-stone">years</span>
                </span>
              </div>
              <div className="stat">
                <span className="stat__label">Cholesterol</span>
                <span className="stat__value">{formatFeatureValue('cholesterol', patient.cholesterol)}</span>
              </div>
            </div>
            <p className="notice mt-6 text-[0.95rem]">{prediction.disclaimer}</p>
          </FramedPanel>
        </div>

        <SHAPChart features={explanation.features} />

        <FramedPanel corners={false} aria-labelledby="whatif-title">
          <h2 id="whatif-title" className="panel__title">
            What if blood pressure changed?
          </h2>
          <p className="panel__note">Pick a target systolic pressure to see how the predicted risk would move.</p>
          <div className="rule" />

          <div className="flex flex-wrap items-end gap-4">
            <div>
              <label htmlFor="target-bp" className="field-label">
                Target systolic pressure <span className="field-hint">(mmHg)</span>
              </label>
              <input
                id="target-bp"
                type="number"
                min={60}
                max={250}
                value={simulatedSysBP}
                onChange={(e) => setSimulatedSysBP(Number(e.target.value))}
                className="field w-40"
              />
            </div>
            <OrnateButton type="button" tone="crimson" small onClick={handleSimulate} disabled={simLoading}>
              {simLoading ? 'Simulating…' : 'Run simulation'}
            </OrnateButton>
          </div>

          {simError && (
            <p className="notice notice--error mt-5" role="alert">
              {simError}
            </p>
          )}

          <WhatIfComparison data={whatIfData} diastolic={whatIfDiastolic} />
        </FramedPanel>

        <ReportAndBatch
          patient={patient}
          riskPercentage={prediction.risk_percentage}
          category={prediction.risk_category}
          bmi={prediction.bmi}
        />
      </div>
    </div>
  );
}
