import { useEffect, useState } from 'react';
import { checkHealth } from '../api/client';
import { CheckCircle2, AlertTriangle } from 'lucide-react';

export default function ServiceStatus() {
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    const verify = async () => {
      try {
        await checkHealth();
        setOnline(true);
      } catch {
        setOnline(false);
      }
    };
    verify();
    const interval = setInterval(verify, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="fixed bottom-4 right-4 z-50 flex items-center space-x-2 px-3 py-1.5 bg-[#FFFBF8] border border-[#E8DFD9] rounded-full shadow-md text-xs font-medium">
      {online === null ? (
        <span className="text-gray-400">Checking API...</span>
      ) : online ? (
        <>
          <CheckCircle2 className="h-4 w-4 text-[#6B9E7E]" />
          <span className="text-gray-700">API Live (127.0.0.1:8000)</span>
        </>
      ) : (
        <>
          <AlertTriangle className="h-4 w-4 text-[#A84040]" />
          <span className="text-[#A84040]">Backend Offline</span>
        </>
      )}
    </div>
  );
}