import { PageBackdrop, PageHeader } from '../components/Backdrop';
import { FramedPanel, OrnateLink } from '../components/ornaments';

// Figures from backend/models/model_metadata.json and models/calibration_report.json
const STATS = [
  { value: '73.2%', label: 'Accuracy on 14,000 test patients' },
  { value: '0.798', label: 'ROC-AUC' },
  { value: '1.5%', label: 'Expected calibration error' },
  { value: '70,000', label: 'Patient records in the dataset' },
];

const SECTIONS = [
  {
    title: 'The model',
    body: [
      'An XGBoost classifier trained on the Kaggle Cardiovascular Disease dataset. It beat logistic regression and random forest baselines.',
      'On held-out test patients it reaches 73.2% accuracy, 75.1% precision, 69.5% recall and a ROC-AUC of 0.798.',
    ],
  },
  {
    title: 'Explanations',
    body: [
      'Every prediction comes with exact SHAP values for that patient, computed by XGBoost’s built-in TreeSHAP. Together with the baseline they add up to the prediction itself.',
      'Across all patients, systolic blood pressure is by far the biggest factor, followed by age and cholesterol.',
    ],
  },
  {
    title: 'Calibration',
    body: [
      'The percentages are well calibrated: a predicted 30% risk means roughly 30% of similar patients had the disease (Brier score 0.182).',
      'The top end runs slightly high. Patients predicted at about 92% had an 85% disease rate, so very high scores are best read as “high”. Platt scaling was tested and not deployed because the gain was small.',
    ],
  },
  {
    title: 'Inputs and stack',
    body: [
      'Eleven inputs: age, gender, height, weight, systolic and diastolic blood pressure, cholesterol, glucose, smoking, alcohol intake and physical activity.',
      'FastAPI with Pydantic validation on Render, and a React, TypeScript, Vite and Tailwind frontend on Vercel. The data preparation is also rebuilt in SQL with DuckDB.',
    ],
  },
];

export default function About() {
  return (
    <div className="page">
      <PageBackdrop />
      <PageHeader
        title="About the model"
        intro="How CardioLens turns eleven basic health details into a risk score, and how it explains every result."
      />

      <div className="mx-auto max-w-5xl space-y-7 px-5">
        <FramedPanel aria-label="Model performance" className="py-12">
          <dl className="grid grid-cols-2 gap-y-6 text-center lg:grid-cols-4">
            {STATS.map((s) => (
              <div key={s.label} className="flex flex-col px-3">
                <dt className="order-2 mt-1 text-[0.95rem] text-stone">{s.label}</dt>
                <dd className="font-display text-[2.4rem] font-semibold leading-none text-cream">{s.value}</dd>
              </div>
            ))}
          </dl>
        </FramedPanel>

        <div className="grid gap-7 md:grid-cols-2">
          {SECTIONS.map((section) => (
            <FramedPanel key={section.title} corners={false} aria-labelledby={`about-${section.title}`}>
              <h2 id={`about-${section.title}`} className="panel__title">
                {section.title}
              </h2>
              <div className="rule" />
              <div className="space-y-3 leading-relaxed text-stone">
                {section.body.map((p) => (
                  <p key={p}>{p}</p>
                ))}
              </div>
            </FramedPanel>
          ))}
        </div>

        <div className="notice text-center">
          A research and learning project, not a medical device. Do not use it to diagnose or treat anyone.
        </div>

        <div className="flex justify-center pt-2">
          <OrnateLink to="/analyze" tone="crimson">
            Launch Clinical Analysis
          </OrnateLink>
        </div>
      </div>
    </div>
  );
}
