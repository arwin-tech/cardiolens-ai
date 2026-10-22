import { useEffect, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { SHAPFeature } from '../types/api';
import { FEATURE_LABELS, formatFeatureValue } from '../lib/labels';
import { FramedPanel } from './ornaments';

interface Props {
  features: SHAPFeature[];
}

type Row = SHAPFeature & { label: string };

function ShapTooltip({ active, payload }: { active?: boolean; payload?: { payload: Row }[] }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="rounded-sm border border-gold/40 bg-ink-2 px-3 py-2 text-[0.95rem] shadow-xl">
      <div className="font-medium text-cream">{row.label}</div>
      <div className="text-stone">Patient value: {formatFeatureValue(row.feature, row.value)}</div>
      <div className="text-stone">
        SHAP: {row.shap_value > 0 ? '+' : ''}
        {row.shap_value.toFixed(3)} ({row.shap_value > 0 ? 'raises' : 'lowers'} risk)
      </div>
    </div>
  );
}

function useNarrow() {
  const query = '(max-width: 640px)';
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const onChange = () => setNarrow(mq.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);
  return narrow;
}

export default function SHAPChart({ features }: Props) {
  const narrow = useNarrow();
  const data: Row[] = [...features]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .slice(0, 8)
    .map((f) => ({ ...f, label: FEATURE_LABELS[f.feature] ?? f.feature }));

  return (
    <FramedPanel corners={false} aria-labelledby="shap-title">
      <h2 id="shap-title" className="panel__title">
        What drove this score
      </h2>
      <p className="panel__note">
        Exact SHAP values for this patient, in log-odds. Bars to the right raised the predicted risk; bars to the left
        lowered it.
      </p>
      <div className="rule" />

      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: narrow ? 8 : 24, bottom: 4, left: 0 }}>
            <CartesianGrid horizontal={false} stroke="rgba(183,154,108,0.12)" />
            <XAxis
              type="number"
              tick={{ fill: '#a8a199', fontSize: 13 }}
              axisLine={{ stroke: 'rgba(183,154,108,0.35)' }}
              tickLine={false}
              tickFormatter={(v: number) => v.toFixed(2)}
            />
            <YAxis
              dataKey="label"
              type="category"
              width={narrow ? 96 : 128}
              tick={{ fill: '#dcd3c6', fontSize: narrow ? 12 : 14 }}
              axisLine={false}
              tickLine={false}
            />
            <ReferenceLine x={0} stroke="rgba(228,204,159,0.55)" />
            <Tooltip cursor={{ fill: 'rgba(183,154,108,0.08)' }} content={<ShapTooltip />} />
            <Bar dataKey="shap_value" barSize={18} radius={2}>
              {data.map((row) => (
                <Cell key={row.feature} fill={row.shap_value > 0 ? '#B3261E' : '#5E9E6E'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 flex flex-wrap gap-x-6 gap-y-2 text-[0.95rem] text-stone">
        <span className="inline-flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-crimson-ember" aria-hidden="true" /> Raises risk
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full bg-verdigris" aria-hidden="true" /> Lowers risk
        </span>
      </div>
    </FramedPanel>
  );
}
