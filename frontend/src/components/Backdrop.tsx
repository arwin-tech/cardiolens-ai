import { ReactNode } from 'react';
import columns from '../assets/columns-left.webp';
import column from '../assets/column-single.webp';
import { Divider } from './ornaments';

/* The hero's temple: faint arcade behind the title, a colonnade between the
   cards and a perspective floor, drawn in the mockup's own coordinates. */
const FLOOR_Y = [250, 261, 274, 290, 309, 333, 362, 398, 443, 500];
const FLOOR_RAYS = Array.from({ length: 21 }, (_, i) => -1400 + i * 190);
const COLONNADE_X = [385, 639];

export function HeroBackdrop() {
  return (
    <div className="hero__backdrop" aria-hidden="true">
      <div className="hero__glow" />
      <svg className="hero__architecture" viewBox="0 0 1024 504" fill="none">
        <defs>
          <linearGradient id="floor-fade" x1="0" y1="250" x2="0" y2="504" gradientUnits="userSpaceOnUse">
            <stop offset="0" stopColor="#fff" stopOpacity="1" />
            <stop offset="0.75" stopColor="#fff" stopOpacity="0.6" />
            <stop offset="1" stopColor="#fff" stopOpacity="0" />
          </linearGradient>
          <mask id="floor-mask" maskUnits="userSpaceOnUse" x="0" y="250" width="1024" height="254">
            <rect x="0" y="250" width="1024" height="254" fill="url(#floor-fade)" />
          </mask>
        </defs>

        <g stroke="rgba(222,206,176,0.075)" strokeWidth="1">
          <path d="M372 280V150a140 140 0 0 1 280 0v130" />
          <path d="M384 280V150a128 128 0 0 1 256 0v130" />
          <path d="M228 280V180a72 72 0 0 1 144 0" />
          <path d="M240 280V180a60 60 0 0 1 120 0" />
          <path d="M652 180a72 72 0 0 1 144 0v100" />
          <path d="M664 180a60 60 0 0 1 120 0v100" />
          <path d="M364 150h28M632 150h28M220 180h28M776 180h28" />
        </g>

        <g stroke="rgba(214,198,168,0.085)" strokeWidth="1">
          {COLONNADE_X.map((x) => (
            <g key={x}>
              <path d={`M${x - 15} 168h30M${x - 13} 175h26`} />
              <path d={`M${x - 11} 175v105M${x + 11} 175v105M${x - 5.5} 178v100M${x} 178v100M${x + 5.5} 178v100`} />
              <path d={`M${x - 16} 280h32v12h-32zM${x - 19} 292h38v6h-38z`} />
            </g>
          ))}
        </g>

        <g mask="url(#floor-mask)" stroke="rgba(214,198,168,0.07)" strokeWidth="1">
          {FLOOR_Y.map((y) => (
            <path key={y} d={`M0 ${y}H1024`} />
          ))}
          {FLOOR_RAYS.map((x) => {
            // ray from the vanishing point (512, 140), clipped to the floor (y >= 250)
            const t = (250 - 140) / (504 - 140);
            const x0 = 512 + (x - 512) * t;
            return <path key={x} d={`M${x0.toFixed(1)} 250L${x} 504`} />;
          })}
        </g>
      </svg>
      <img className="hero__columns-left" src={columns} alt="" />
      <img className="hero__column-right" src={column} alt="" />
    </div>
  );
}

export function PageBackdrop() {
  return (
    <div className="page-backdrop" aria-hidden="true">
      <div className="page-backdrop__glow" />
      <img className="page-backdrop__columns" src={columns} alt="" />
      <img className="page-backdrop__columns page-backdrop__columns--right" src={columns} alt="" />
    </div>
  );
}

export function PageHeader({ title, intro, action }: { title: string; intro?: ReactNode; action?: ReactNode }) {
  return (
    <header className="page-head">
      <h1 className="page-head__title">{title}</h1>
      <Divider />
      {intro && <p className="page-head__intro">{intro}</p>}
      {action && <div className="mt-6 flex justify-center">{action}</div>}
    </header>
  );
}
