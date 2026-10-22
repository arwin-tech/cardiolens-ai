import { createContext, ReactNode, useContext, useEffect, useRef, useState } from 'react';
import { healthCheck } from '../api/client';

// 'waking' = the free Render server is starting up (it sleeps when idle; up to ~1 minute)
export type BackendState = 'checking' | 'waking' | 'online' | 'offline';

const BackendStatusContext = createContext<BackendState>('checking');

export function BackendStatusProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<BackendState>('checking');
  const failingSince = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    // No answer within 4 seconds usually means the server is waking up, not broken
    const slow = setTimeout(() => {
      if (!cancelled) setState((s) => (s === 'checking' ? 'waking' : s));
    }, 4000);

    const check = async () => {
      const res = await healthCheck(15000).catch(() => null);
      if (cancelled) return;
      if (res && res.ok) {
        failingSince.current = null;
        setState('online');
        timer = setTimeout(check, 30000);
      } else {
        if (failingSince.current === null) failingSince.current = Date.now();
        const waitedMs = Date.now() - failingSince.current;
        setState(waitedMs > 120000 ? 'offline' : 'waking');
        timer = setTimeout(check, 5000);
      }
    };

    check(); // also wakes the server as soon as anyone opens the site
    return () => {
      cancelled = true;
      clearTimeout(timer);
      clearTimeout(slow);
    };
  }, []);

  return <BackendStatusContext.Provider value={state}>{children}</BackendStatusContext.Provider>;
}

export const useBackendState = () => useContext(BackendStatusContext);
