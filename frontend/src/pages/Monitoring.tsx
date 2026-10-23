import { useCallback, useEffect, useState } from 'react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { PageBackdrop, PageHeader } from '../components/Backdrop';
import { FramedPanel, OrnateButton } from '../components/ornaments';
import { getMonitoring } from '../api/client';
import { useBackendState } from '../context/BackendStatus';
import { FEATURE_LABELS } from '../lib/labels';
import { MonitoringSummary } from '../types/api';

const MODEL_NAMES: Record<string, string> = { kaggle: 'Clinic model', nhanes: 'US population model' };
const STATUS_CLASS: Record<string, string> = {
  stable: 'pill pill--ok',
  'moderate shift': 'pill pill--warn',
  'major shift': 'pill pill--bad',
};

const count = (n: number | null | undefined) => (n === null || n === undefined ? '<5' : n.toLocaleString());

export default function Monitoring() {
  const backend = useBackendState();
  const [data, setData] = useState<MonitoringSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    getMonitoring(30)
      .then(setData)
      .catch((e: Error) => setError(e.message || 'Could not load monitoring data.'))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (backend === 'online') load();
  }, [backend, load]);

  return (
    <div className="page">
      <PageBackdrop />
      <PageHeader
        title="Live monitoring"
        intro="How the deployed models are being used over the last 30 days, recorded without storing any patient data."
        action={
          <OrnateButton type="button" tone="umber" small onClick={load} disabled={loading || backend !== 'online'}>
            {loading ? 'Refreshing…' : 'Refresh'}
          </OrnateButton>
        }
      />

      <div className="mx-auto max-w-5xl space-y-7 px-5">
        {backend !== 'online' && !data && <div className="notice">Waiting for the server to wake up…</div>}
        {error && (
          <div className="notice notice--error" role="alert">
            {error}
          </div>
        )}

        {data && (
          <>
            {data.storage === 'in_memory' && (
              <div className="notice">
                Counts are kept in the server’s memory, so they reset whenever the free server restarts or goes to
                sleep. Connecting a database keeps them permanently.
              </div>
            )}

            <div className="grid gap-4 sm:grid-cols-3">
              <div className="stat">
                <span className="stat__label">Predictions</span>
                <span className="stat__value">{data.predictions_total.toLocaleString()}</span>
              </div>
              {Object.keys(MODEL_NAMES).map((id) => (
                <div className="stat" key={id}>
                  <span className="stat__label">{MODEL_NAMES[id]}</span>
                  <span className="stat__value">{(data.predictions_by_model[id] ?? 0).toLocaleString()}</span>
                </div>
              ))}
            </div>

            <FramedPanel corners={false} aria-labelledby="per-day-title">
              <h2 id="per-day-title" className="panel__title">
                Predictions per day
              </h2>
              <div className="rule" />
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.predictions_per_day.map((d) => ({ ...d, label: d.day.slice(5) }))}>
                    <CartesianGrid vertical={false} stroke="rgba(183,154,108,0.12)" />
                    <XAxis dataKey="label" tick={{ fill: '#a8a199', fontSize: 12 }} interval={4} tickLine={false} />
                    <YAxis allowDecimals={false} tick={{ fill: '#a8a199', fontSize: 12 }} width={36} axisLine={false} tickLine={false} />
                    <Tooltip
                      cursor={{ fill: 'rgba(183,154,108,0.08)' }}
                      contentStyle={{ background: '#171615', border: '1px solid rgba(183,154,108,0.4)', color: '#f1e6d3' }}
                      labelStyle={{ color: '#dcd3c6' }}
                    />
                    <Bar dataKey="predictions" fill="#B79A6C" radius={2} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </FramedPanel>

            <div className="grid gap-7 md:grid-cols-2">
              <FramedPanel corners={false} aria-labelledby="latency-title">
                <h2 id="latency-title" className="panel__title">
                  Response times
                </h2>
                <div className="rule" />
                {Object.keys(data.response_times).length === 0 ? (
                  <p className="text-stone">No requests yet.</p>
                ) : (
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Endpoint</th>
                        <th>Requests</th>
                        <th>Median</th>
                        <th>95th percentile</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(data.response_times).map(([endpoint, s]) => (
                        <tr key={endpoint}>
                          <td className="text-parchment">{endpoint}</td>
                          <td>{s.requests}</td>
                          <td>{s.median ?? '—'}</td>
                          <td>{s.p95 ?? '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </FramedPanel>

              <FramedPanel corners={false} aria-labelledby="mix-title">
                <h2 id="mix-title" className="panel__title">
                  Risk mix
                </h2>
                <div className="rule" />
                {Object.keys(data.risk_categories).length === 0 ? (
                  <p className="text-stone">No predictions yet.</p>
                ) : (
                  Object.entries(data.risk_categories).map(([id, cats]) => (
                    <div key={id} className="mb-4 last:mb-0">
                      <p className="mb-1 font-medium text-parchment">{MODEL_NAMES[id] ?? id}</p>
                      <p className="text-stone">
                        {['Low', 'Moderate', 'Elevated', 'High']
                          .filter((c) => c in cats)
                          .map((c) => `${c}: ${count(cats[c])}`)
                          .join(', ')}
                      </p>
                    </div>
                  ))
                )}
              </FramedPanel>
            </div>

            <FramedPanel corners={false} aria-labelledby="drift-title">
              <h2 id="drift-title" className="panel__title">
                Input drift
              </h2>
              <p className="panel__note">
                Population stability index (PSI) between recent inputs and each model’s training data. Under 0.1 is
                stable; over 0.25 means the inputs look clearly different.
              </p>
              <div className="rule" />
              {Object.entries(data.drift).map(([id, d]) => (
                <div key={id} className="mb-6 last:mb-0">
                  <p className="mb-2 font-medium text-parchment">{MODEL_NAMES[id] ?? id}</p>
                  {d.status === 'not_enough_data' ? (
                    <p className="text-stone">
                      Needs {d.needed} predictions to compare fairly (has {d.have}).
                    </p>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Input</th>
                            <th>PSI</th>
                            <th>Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(d.features)
                            .sort((a, b) => b[1].psi - a[1].psi)
                            .map(([feature, f]) => (
                              <tr key={feature}>
                                <td className="text-parchment">{FEATURE_LABELS[feature] ?? feature}</td>
                                <td>{f.psi.toFixed(3)}</td>
                                <td>
                                  <span className={STATUS_CLASS[f.status] ?? 'pill'}>{f.status}</span>
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              ))}
            </FramedPanel>

            <FramedPanel aria-labelledby="privacy-title">
              <h2 id="privacy-title" className="panel__title">
                What is recorded
              </h2>
              <div className="rule" />
              <div className="space-y-3 leading-relaxed text-stone">
                <p>Only {data.privacy.stored}.</p>
                <p>
                  Any count below {data.privacy.smallest_count_shown} is hidden, and drift is only calculated once a
                  model has {data.privacy.drift_needs_predictions} predictions, so a quiet day can never reveal one
                  person’s inputs.
                </p>
              </div>
            </FramedPanel>
          </>
        )}
      </div>
    </div>
  );
}
