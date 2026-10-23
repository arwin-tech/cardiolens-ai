import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import PatientForm from '../components/PatientForm';
import { PageBackdrop, PageHeader } from '../components/Backdrop';
import { runAssessment } from '../lib/assess';
import { ModelId, ModelInfo, PatientInput } from '../types/api';
import { getModels } from '../api/client';
import { useBackendState } from '../context/BackendStatus';

export default function Analyze() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const backend = useBackendState();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState<ModelId>('kaggle');

  // Offer the US population model only when the server has it
  useEffect(() => {
    if (backend !== 'online') return;
    getModels()
      .then((res) => setModels(res.models))
      .catch(() => setModels([]));
  }, [backend]);
  const waking = backend !== 'online';

  const handleFormSubmit = async (data: PatientInput) => {
    setLoading(true);
    setError(null);
    try {
      await runAssessment(data, model);
      navigate('/results');
    } catch (err: any) {
      setError(err.message || 'Failed to communicate with FastAPI server. Ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <PageBackdrop />
      <PageHeader
        title="Run a risk assessment"
        intro="Enter a patient's details or start from an example. You'll get a risk score and a breakdown of the factors behind it."
      />

      <div className="mx-auto max-w-4xl space-y-6 px-5">
        {waking && !error && (
          <div className="notice">
            The demo runs on a free server that sleeps when idle, so it can take up to a minute to wake up. You can fill
            in the form or pick an example now, and your result will appear as soon as it's ready.
          </div>
        )}

        {error && (
          <div className="notice notice--error" role="alert">
            {error}
          </div>
        )}

        <PatientForm
          onSubmit={handleFormSubmit}
          loading={loading}
          waking={waking}
          models={models}
          model={model}
          onModelChange={setModel}
        />
      </div>
    </div>
  );
}
