import { ReactNode } from 'react';

export default function ArchCard({ art, title, children }: { art: string; title: string; children: ReactNode }) {
  return (
    <div className="arch-cell">
      <article className="arch-card">
        <div className="arch-card__art">
          <img src={art} alt="" />
        </div>
        <h3 className="arch-card__title">{title}</h3>
        <p className="arch-card__text">{children}</p>
      </article>
    </div>
  );
}
