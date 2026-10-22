import { useBackendState, BackendState } from '../context/BackendStatus';
import { PlaqueCap } from './ornaments';

const LABELS: Record<BackendState, string> = {
  checking: 'Connecting to the model…',
  waking: 'Waking up the model (up to a minute)…',
  online: 'Backend Online',
  offline: 'Backend unreachable, retrying…',
};

export default function ServiceStatus() {
  const state = useBackendState();

  return (
    <div className="status-plaque" role="status" aria-live="polite">
      <PlaqueCap className="status-plaque__cap status-plaque__cap--left" />
      <PlaqueCap className="status-plaque__cap status-plaque__cap--right" />
      <span className={`gem gem--${state}`} aria-hidden="true" />
      <span className="status-plaque__label">{LABELS[state]}</span>
    </div>
  );
}
