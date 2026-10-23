import { useEffect, useState } from 'react';
import { riskColor } from '../lib/labels';

interface Props {
  percentage: number;
  category: string;
  range?: [number, number] | null;
}

const R = 74;
const CIRC = 2 * Math.PI * R;
const TICKS = Array.from({ length: 40 }, (_, i) => i);

/** The engraved brass rings and tick marks shared by the live and empty dials. */
function DialFace() {
  return (
    <>
      <circle cx="100" cy="100" r="97" fill="none" stroke="rgba(183,154,108,0.45)" strokeWidth="1" />
      <circle cx="100" cy="100" r="92.5" fill="none" stroke="rgba(183,154,108,0.16)" strokeWidth="1" />
      {TICKS.map((i) => {
        const a = (i / 40) * Math.PI * 2 - Math.PI / 2;
        const major = i % 10 === 0;
        const r1 = major ? 83 : 86;
        return (
          <line
            key={i}
            x1={100 + r1 * Math.cos(a)}
            y1={100 + r1 * Math.sin(a)}
            x2={100 + 90 * Math.cos(a)}
            y2={100 + 90 * Math.sin(a)}
            stroke={major ? 'rgba(228,204,159,0.85)' : 'rgba(183,154,108,0.45)'}
            strokeWidth={major ? 1.4 : 1}
          />
        );
      })}
      <circle cx="100" cy="100" r={R} fill="none" stroke="#2a2621" strokeWidth="11" />
    </>
  );
}

const innerDisc = <circle cx="100" cy="100" r="66" fill="rgba(10,9,8,0.6)" stroke="rgba(183,154,108,0.18)" strokeWidth="1" />;

/** Risk shown as an engraved brass dial: the arc sweeps to the predicted probability. */
export default function RiskGauge({ percentage, category, range }: Props) {
  const pct = Math.max(0, Math.min(100, percentage));
  const color = riskColor(category);
  const [shown, setShown] = useState(0);

  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(pct));
    return () => cancelAnimationFrame(id);
  }, [pct]);

  return (
    <div className="flex flex-col items-center text-center">
      <h2 className="panel__title">Cardiovascular risk</h2>
      <div className="relative mt-5 h-56 w-56">
        <svg viewBox="0 0 200 200" className="h-full w-full" role="img" aria-label={`Predicted risk ${pct.toFixed(1)} percent, ${category}`}>
          <DialFace />
          <circle
            cx="100"
            cy="100"
            r={R}
            fill="none"
            stroke={color}
            strokeWidth="11"
            strokeDasharray={`${(shown / 100) * CIRC} ${CIRC}`}
            transform="rotate(-90 100 100)"
            style={{ transition: 'stroke-dasharray 1.1s cubic-bezier(0.3, 0.7, 0.2, 1)', filter: `drop-shadow(0 0 5px ${color}88)` }}
          />
          {innerDisc}
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-display text-[2.7rem] font-semibold leading-none text-cream">{pct.toFixed(1)}%</span>
          <span className="mt-2 font-medium" style={{ color }}>
            {category} risk
          </span>
        </div>
      </div>
      {range && (
        <p className="mt-5 text-parchment">
          Calibrated range{' '}
          <strong className="font-semibold text-cream">
            {range[0].toFixed(1)}–{range[1].toFixed(1)}%
          </strong>
        </p>
      )}
      <p className={`${range ? 'mt-2' : 'mt-5'} max-w-[19rem] text-[0.95rem] text-stone`}>
        {range
          ? 'The model’s estimate, and how often held-out patients with similar scores actually had heart disease (Venn–Abers).'
          : 'The model’s estimated probability that this patient has cardiovascular disease.'}
      </p>
    </div>
  );
}

/** The same dial with no reading yet, for the empty Results page. */
export function EmptyDial() {
  return (
    <div className="relative mx-auto h-44 w-44" aria-hidden="true">
      <svg viewBox="0 0 200 200" className="h-full w-full opacity-80">
        <DialFace />
        {innerDisc}
      </svg>
      <div className="absolute inset-0 flex items-center justify-center font-display text-[2.6rem] leading-none text-stone-2">
        —
      </div>
    </div>
  );
}
