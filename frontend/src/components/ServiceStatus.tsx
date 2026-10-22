import { CheckCircle2, AlertTriangle, Loader2 } from 'lucide-react';
import { useBackendState } from '../context/BackendStatus';

export default function ServiceStatus() {
  const state = useBackendState();

  return (
    <div className="fixed bottom-4 right-4 z-50 flex items-center space-x-2 px-3 py-1.5 bg-[#FFFBF8] border border-[#E8DFD9] rounded-full shadow-md text-xs font-medium">
      {state === 'checking' && <span className="text-gray-400">Checking API...</span>}
      {state === 'waking' && (
        <>
          <Loader2 className="h-4 w-4 text-[#8B2D2D] animate-spin" />
          <span className="text-gray-700">Waking up the model (up to a minute)...</span>
        </>
      )}
      {state === 'online' && (
        <>
          <CheckCircle2 className="h-4 w-4 text-[#6B9E7E]" />
          <span className="text-gray-700">Backend Online</span>
        </>
      )}
      {state === 'offline' && (
        <>
          <AlertTriangle className="h-4 w-4 text-[#A84040]" />
          <span className="text-[#A84040]">Backend unreachable, retrying...</span>
        </>
      )}
    </div>
  );
}
