import { useEffect, useState } from 'react';
import { getCounterfactuals } from '../api/client';
import { CounterfactualResponse, ModelId, PatientInput } from '../types/api';
import { FramedPanel } from './ornaments';

/** The smallest sets of changes that lower this patient's predicted risk. */
export default function CounterfactualPanel({ patient, model }: { patient: PatientInput; model: ModelId }) {
  const [data, setData] = useState<CounterfactualResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    getCounterfactuals(patient, model)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message || 'Could not load suggestions.');
      });
    return () => {
      cancelled = true;
    };
  }, [patient, model]);

  return (
    <FramedPanel corners={false} aria-labelledby="cf-title">
      <h2 id="cf-title" className="panel__title">
        Smallest changes that lower this risk
      </h2>
      <p className="panel__note">
        Found by trying every combination of realistic, healthier changes. Age, sex, height and medication stay fixed.
      </p>
      <div className="rule" />

      {!data && !error && <p className="text-stone">Searching for changes…</p>}
      {error && (
        <p className="notice notice--error" role="alert">
          {error}
        </p>
      )}

      {data && data.plans.length === 0 && (
        <p className="text-stone">
          No healthier changes are available for these inputs, so there is nothing to suggest.
        </p>
      )}

      {data && data.plans.length > 0 && (
        <ol className="plans">
          {data.plans.map((plan) => (
            <li key={plan.changes.map((c) => c.label).join('|')} className="plan">
              <span className="plan__count">
                {plan.changes.length} {plan.changes.length === 1 ? 'change' : 'changes'}
              </span>
              <span className="plan__changes">
                {plan.changes.map((c) => (
                  <span key={c.feature} className="tag">
                    {c.label}
                  </span>
                ))}
              </span>
              <span className="plan__result">
                <span className="plan__risk">{plan.risk.toFixed(1)}%</span>
                <span className="plan__delta">{plan.delta.toFixed(1)} points</span>
              </span>
            </li>
          ))}
        </ol>
      )}

      {data && <p className="mt-5 text-[0.95rem] text-stone-2">{data.note}</p>}
    </FramedPanel>
  );
}
