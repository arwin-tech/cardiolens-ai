import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import PatientForm from '../components/PatientForm';
import { predictRisk, explainPrediction } from '../api/client';
import { PatientInput } from '../types/api';

export default function Analyze() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const handleFormSubmit = async (data: PatientInput) => {
    setLoading(true);
    setError(null);
    try {
      const prediction = await predictRisk(data);
      const explanation = await explainPrediction(data);
      
      sessionStorage.setItem('lastPrediction', JSON.stringify(prediction));
      sessionStorage.setItem('lastExplanation', JSON.stringify(explanation));

      navigate('/results');
    } catch (err: any) {
      setError(err.message || 'Failed to communicate with FastAPI server. Ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-6 py-10 space-y-8">
      <div>
        <h2 className="text-3xl font-extrabold text-[#8B2D2D]">Run Risk Assessment</h2>
        <p className="text-sm text-gray-600 mt-1">Enter patient clinical metrics below to execute model evaluation.</p>
      </div>

      {error && (
        <div className="p-4 bg-[#A84040]/10 border border-[#A84040] text-[#A84040] text-sm rounded-lg">
          {error}
        </div>
      )}

      <PatientForm onSubmit={handleFormSubmit} loading={loading} />
    </div>
  );
}