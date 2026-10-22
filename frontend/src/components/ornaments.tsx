import { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from 'react';
import { Link } from 'react-router-dom';
import filigree from '../assets/filigree-corner.webp';

/** Small gold scroll drawn over each corner of a button frame. */
function Flourish({ corner }: { corner: 'tl' | 'tr' | 'bl' | 'br' }) {
  return (
    <svg className={`flourish flourish--${corner}`} viewBox="0 0 15 15" fill="none" aria-hidden="true">
      <path d="M1.2 13.8V7.2C1.2 3.9 3.9 1.2 7.2 1.2h6.6" stroke="currentColor" strokeWidth="1.1" strokeLinecap="round" />
      <path d="M4.3 10.8V8.4c0-2.3 1.8-4.1 4.1-4.1h2.4" stroke="currentColor" strokeWidth="0.9" strokeLinecap="round" opacity="0.75" />
      <circle cx="4.3" cy="4.3" r="1.15" fill="currentColor" />
    </svg>
  );
}

export function Flourishes() {
  return (
    <>
      <Flourish corner="tl" />
      <Flourish corner="tr" />
      <Flourish corner="bl" />
      <Flourish corner="br" />
    </>
  );
}

type Tone = 'crimson' | 'umber';

function btnClass(tone: Tone, small?: boolean, block?: boolean, extra = '') {
  return ['ornate-btn', `ornate-btn--${tone}`, small && 'ornate-btn--small', block && 'ornate-btn--block', extra]
    .filter(Boolean)
    .join(' ');
}

export function OrnateLink({
  to,
  tone = 'crimson',
  small,
  className,
  children,
}: {
  to: string;
  tone?: Tone;
  small?: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <Link to={to} className={btnClass(tone, small, false, className)}>
      <Flourishes />
      {children}
    </Link>
  );
}

export function OrnateButton({
  tone = 'crimson',
  small,
  block,
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { tone?: Tone; small?: boolean; block?: boolean }) {
  return (
    <button className={btnClass(tone, small, block, className)} {...rest}>
      <Flourishes />
      {children}
    </button>
  );
}

/** The gold heartbeat glyph from the brand kit. */
export function PulseGlyph({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 64 32" fill="none" aria-hidden="true">
      <polyline
        points="1,19 15,19 18.5,13 22,19 25,22 29.5,3 35,29 38.5,15 41.5,19.5 45,12 48,19 63,19"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function Divider() {
  return (
    <div className="divider" aria-hidden="true">
      <PulseGlyph />
    </div>
  );
}

/** Dark panel with a gold hairline frame; `corners` adds the engraved filigree. */
export function FramedPanel({
  corners = true,
  className = '',
  children,
  ...rest
}: HTMLAttributes<HTMLElement> & { corners?: boolean }) {
  return (
    <section className={`panel ${corners ? '' : 'panel--plain'} ${className}`} {...rest}>
      {corners && (
        <>
          <img src={filigree} alt="" aria-hidden="true" className="panel__corner panel__corner--tl" />
          <img src={filigree} alt="" aria-hidden="true" className="panel__corner panel__corner--tr" />
          <img src={filigree} alt="" aria-hidden="true" className="panel__corner panel__corner--bl" />
          <img src={filigree} alt="" aria-hidden="true" className="panel__corner panel__corner--br" />
        </>
      )}
      {children}
    </section>
  );
}

/** Ornamental end cap for the status plaque (mirrored for the right side in CSS). */
export function PlaqueCap({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 13 30" fill="none" aria-hidden="true">
      <path d="M12.5 3.4C7.6 4.6 5.6 9.2 1.7 15c3.9 5.8 5.9 10.4 10.8 11.6" stroke="currentColor" strokeWidth="1.2" />
      <path d="M12.5 7.3C9.5 8.5 7.9 11.3 5.9 15c2 3.7 3.6 6.5 6.6 7.7" stroke="currentColor" strokeWidth="0.9" opacity="0.7" />
      <circle cx="1.9" cy="15" r="1.5" fill="currentColor" />
      <path d="M9.1 5.1C8.1 2.3 5.3 1.8 4.2 3.4c-.9 1.4.5 2.8 1.9 2" stroke="currentColor" strokeWidth="0.9" />
      <path d="M9.1 24.9c-1 2.8-3.8 3.3-4.9 1.7-.9-1.4.5-2.8 1.9-2" stroke="currentColor" strokeWidth="0.9" />
    </svg>
  );
}
