export default function About() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-10 space-y-8">
      <div>
        <h2 className="text-3xl font-extrabold text-[#8B2D2D]">About CardioLens AI Architecture</h2>
        <p className="text-sm text-gray-600 mt-1">Machine Learning Pipeline & Clinical Explainability Standard</p>
      </div>

      <div className="p-8 bg-[#FFFBF8] border border-[#E8DFD9] rounded-medical shadow-sm space-y-6 text-sm text-gray-700 leading-relaxed">
        <h3 className="text-xl font-bold text-gray-800">Model Specifications</h3>
        <ul className="list-disc pl-5 space-y-2">
          <li><strong>Algorithm Core:</strong> XGBoost (Extreme Gradient Boosting Classification Pipeline).</li>
          <li><strong>Explainability Layer:</strong> SHAP (SHapley Additive exPlanations) tree-explainer routines.</li>
          <li><strong>Backend Framework:</strong> FastAPI with Pydantic input boundary enforcement.</li>
          <li><strong>Frontend Framework:</strong> React 18 with TypeScript, Vite, and Tailwind CSS.</li>
        </ul>
      </div>
    </div>
  );
}