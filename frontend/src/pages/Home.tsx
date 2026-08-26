import { Link } from 'react-router-dom';
import { ShieldCheck, Cpu, LineChart } from 'lucide-react';

export default function Home() {
  return (
    <div className="max-w-6xl mx-auto px-6 py-12 space-y-16">
      <section className="text-center space-y-6 max-w-3xl mx-auto">
        <h1 className="text-5xl font-extrabold text-[#8B2D2D] tracking-tight leading-tight">
          Explainable AI for Cardiovascular Health Assessment
        </h1>
        <p className="text-lg text-gray-600">
          CardioLens AI combines robust gradient-boosted ML pipelines (XGBoost) with SHAP explainability to evaluate risk scores and counterfactual intervention scenarios.
        </p>
        <div className="flex justify-center space-x-4 pt-4">
          <Link to="/analyze" className="px-6 py-3 bg-[#8B2D2D] hover:bg-[#722323] text-white font-semibold rounded-medical shadow-md transition-all">
            Launch Clinical Analysis
          </Link>
          <Link to="/about" className="px-6 py-3 bg-[#FFFBF8] border border-[#E8DFD9] text-gray-700 hover:text-[#8B2D2D] font-semibold rounded-medical transition-all">
            View Model Architecture
          </Link>
        </div>
      </section>

      <section className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="p-6 bg-[#FFFBF8] border border-[#E8DFD9] rounded-medical shadow-sm space-y-3">
          <Cpu className="h-8 w-8 text-[#8B2D2D]" />
          <h3 className="font-bold text-lg text-gray-800">Gradient Boosted Engine</h3>
          <p className="text-sm text-gray-600">Trained on over 70,000 patient records to provide accurate baseline cardiovascular risk probabilities.</p>
        </div>
        <div className="p-6 bg-[#FFFBF8] border border-[#E8DFD9] rounded-medical shadow-sm space-y-3">
          <LineChart className="h-8 w-8 text-[#8B2D2D]" />
          <h3 className="font-bold text-lg text-gray-800">Local SHAP Explainability</h3>
          <p className="text-sm text-gray-600">Inspect individual feature contribution rankings to see exactly which risk factors drove the output score.</p>
        </div>
        <div className="p-6 bg-[#FFFBF8] border border-[#E8DFD9] rounded-medical shadow-sm space-y-3">
          <ShieldCheck className="h-8 w-8 text-[#8B2D2D]" />
          <h3 className="font-bold text-lg text-gray-800">Counterfactual Analysis</h3>
          <p className="text-sm text-gray-600">Simulate lifestyle or blood pressure changes dynamically to estimate risk delta before clinical intervention.</p>
        </div>
      </section>
    </div>
  );
}