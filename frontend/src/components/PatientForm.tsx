import { useState } from 'react';
import { PatientInput } from '../types/api';

interface Props {
  onSubmit: (data: PatientInput) => void;
  loading: boolean;
}

export default function PatientForm({ onSubmit, loading }: Props) {
  const [formData, setFormData] = useState<PatientInput>({
    age: 52,
    gender: 1,
    height: 165,
    weight: 82,
    ap_hi: 145,
    ap_lo: 90,
    cholesterol: 2,
    gluc: 1,
    smoke: 0,
    alco: 0,
    active: 1,
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: Number(value) }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="bg-[#FFFBF8] border border-[#E8DFD9] p-8 rounded-medical shadow-sm space-y-6">
      <h3 className="text-xl font-bold text-[#8B2D2D] border-b border-[#E8DFD9] pb-3">Patient Clinical Input Parameters</h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-xs font-semibold uppercase text-gray-500 mb-1">Age (Years)</label>
          <input type="number" name="age" min="18" max="100" value={formData.age} onChange={handleChange} required className="w-full p-2.5 bg-white border border-[#E8DFD9] rounded-lg text-sm focus:outline-none focus:border-[#8B2D2D]" />
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase text-gray-500 mb-1">Gender</label>
          <select name="gender" value={formData.gender} onChange={handleChange} className="w-full p-2.5 bg-white border border-[#E8DFD9] rounded-lg text-sm focus:outline-none focus:border-[#8B2D2D]">
            <option value={1}>Female (1)</option>
            <option value={2}>Male (2)</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase text-gray-500 mb-1">Height (cm)</label>
          <input type="number" name="height" min="100" max="250" value={formData.height} onChange={handleChange} required className="w-full p-2.5 bg-white border border-[#E8DFD9] rounded-lg text-sm focus:outline-none focus:border-[#8B2D2D]" />
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase text-gray-500 mb-1">Weight (kg)</label>
          <input type="number" name="weight" min="30" max="300" value={formData.weight} onChange={handleChange} required className="w-full p-2.5 bg-white border border-[#E8DFD9] rounded-lg text-sm focus:outline-none focus:border-[#8B2D2D]" />
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase text-gray-500 mb-1">Systolic BP (ap_hi)</label>
          <input type="number" name="ap_hi" min="60" max="250" value={formData.ap_hi} onChange={handleChange} required className="w-full p-2.5 bg-white border border-[#E8DFD9] rounded-lg text-sm focus:outline-none focus:border-[#8B2D2D]" />
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase text-gray-500 mb-1">Diastolic BP (ap_lo)</label>
          <input type="number" name="ap_lo" min="30" max="180" value={formData.ap_lo} onChange={handleChange} required className="w-full p-2.5 bg-white border border-[#E8DFD9] rounded-lg text-sm focus:outline-none focus:border-[#8B2D2D]" />
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase text-gray-500 mb-1">Cholesterol Level</label>
          <select name="cholesterol" value={formData.cholesterol} onChange={handleChange} className="w-full p-2.5 bg-white border border-[#E8DFD9] rounded-lg text-sm focus:outline-none focus:border-[#8B2D2D]">
            <option value={1}>1: Normal</option>
            <option value={2}>2: Above Normal</option>
            <option value={3}>3: Well Above Normal</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase text-gray-500 mb-1">Glucose Level</label>
          <select name="gluc" value={formData.gluc} onChange={handleChange} className="w-full p-2.5 bg-white border border-[#E8DFD9] rounded-lg text-sm focus:outline-none focus:border-[#8B2D2D]">
            <option value={1}>1: Normal</option>
            <option value={2}>2: Above Normal</option>
            <option value={3}>3: Well Above Normal</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase text-gray-500 mb-1">Smoker</label>
          <select name="smoke" value={formData.smoke} onChange={handleChange} className="w-full p-2.5 bg-white border border-[#E8DFD9] rounded-lg text-sm focus:outline-none focus:border-[#8B2D2D]">
            <option value={0}>No (0)</option>
            <option value={1}>Yes (1)</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase text-gray-500 mb-1">Alcohol Intake</label>
          <select name="alco" value={formData.alco} onChange={handleChange} className="w-full p-2.5 bg-white border border-[#E8DFD9] rounded-lg text-sm focus:outline-none focus:border-[#8B2D2D]">
            <option value={0}>No (0)</option>
            <option value={1}>Yes (1)</option>
          </select>
        </div>

        <div className="md:col-span-2">
          <label className="block text-xs font-semibold uppercase text-gray-500 mb-1">Physical Activity</label>
          <select name="active" value={formData.active} onChange={handleChange} className="w-full p-2.5 bg-white border border-[#E8DFD9] rounded-lg text-sm focus:outline-none focus:border-[#8B2D2D]">
            <option value={1}>Active Lifestyle (1)</option>
            <option value={0}>Sedentary Lifestyle (0)</option>
          </select>
        </div>
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full py-3 bg-[#8B2D2D] hover:bg-[#722323] text-white font-semibold rounded-medical transition-all shadow-sm disabled:opacity-50"
      >
        {loading ? 'Executing Risk Prediction...' : 'Evaluate Risk Score'}
      </button>
    </form>
  );
}