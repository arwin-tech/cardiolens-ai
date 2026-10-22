import heart from '../assets/hero-heart.webp';

// The ECG trace from the mockup, in the mockup's own pixel coordinates (1024-wide canvas).
const ECG_PATH =
  'M750 200L810 200L815 182L822 193L826 200L832 216L843 125L857 264L866 208L870 199L878 200' +
  'L886 172L892 201L897 213L902 203L908 191L913 199L978 200';

export default function HeartECG({ className = '' }: { className?: string }) {
  return (
    <div className={`heart-ecg ${className}`} aria-hidden="true">
      <img className="heart-ecg__img" src={heart} alt="" />
      <svg className="heart-ecg__svg" viewBox="740 70 250 260">
        <path className="heart-ecg__line" d={ECG_PATH} pathLength={1} />
      </svg>
    </div>
  );
}
