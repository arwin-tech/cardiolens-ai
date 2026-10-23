import { PageBackdrop, PageHeader } from '../components/Backdrop';
import { Link } from 'react-router-dom';
import { FramedPanel, OrnateLink } from '../components/ornaments';
import reports from '../data/model_reports.json';

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

// From models/fairness_report.json (src/fairness_check.py)
const FAIRNESS: [string, string, string, string, string][] = [
  ['Women', '9,080', '0.793', '68.6%', '1.6%'],
  ['Men', '4,920', '0.806', '71.1%', '1.4%'],
  ['Under 45', '2,033', '0.837', '57.4%', '1.7%'],
  ['45–49', '1,793', '0.786', '61.1%', '3.8%'],
  ['50–54', '3,732', '0.786', '60.9%', '1.9%'],
  ['55–59', '3,254', '0.750', '67.5%', '1.5%'],
  ['60 and over', '3,188', '0.715', '85.2%', '2.5%'],
];

interface NhanesReport {
  train: { cycles: string[]; patients: number };
  test_cycle: string;
  test: {
    patients: number;
    cvd_rate: number;
    roc_auc: number;
    roc_auc_95ci: [number, number];
    roc_auc_age_alone: number;
    roc_auc_logistic: number;
    mean_predicted_risk: number;
    ece: number;
  };
  ablation_without_medication: { roc_auc: number };
  comparison_age_30_65: { us_population_model: number; clinic_model: number; age_alone: number };
}

const uncertainty = reports.kaggle.uncertainty;
// JSON imports infer number[] for pairs, so convert through unknown
const nhanes = (reports as unknown as { nhanes: NhanesReport | null }).nhanes;
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

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

        <FramedPanel corners={false} aria-labelledby="about-fairness">
          <h2 id="about-fairness" className="panel__title">
            Fairness check
          </h2>
          <div className="rule" />
          <p className="leading-relaxed text-stone">
            Women and men get almost identical performance. Age is where the model differs: the percentages stay well
            calibrated in every age band, but at the 50% cut-off it catches 57% of disease cases in patients under 45
            versus 85% in patients 60 and over, because younger patients get lower scores overall.
          </p>
          <div className="mt-5 overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Group</th>
                  <th>Patients</th>
                  <th>ROC-AUC</th>
                  <th>Cases caught at 50%</th>
                  <th>Calibration error</th>
                </tr>
              </thead>
              <tbody>
                {FAIRNESS.map(([group, n, auc, recall, ece]) => (
                  <tr key={group}>
                    <td className="text-parchment">{group}</td>
                    <td>{n}</td>
                    <td>{auc}</td>
                    <td>{recall}</td>
                    <td>{ece}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-[0.95rem] text-stone-2">From src/fairness_check.py, on the same 14,000 test patients.</p>
        </FramedPanel>

        <FramedPanel corners={false} aria-labelledby="about-external">
          <h2 id="about-external" className="panel__title">
            Tested on a different population
          </h2>
          <div className="rule" />
          <div className="space-y-3 leading-relaxed text-stone">
            <p>
              The model was also tested, unchanged, on 2,916 US adults from NHANES 2013–2014, the CDC’s national
              health survey. It does not transfer well: ROC-AUC fell from 0.798 to 0.631, and ranking people by age alone
              (0.696) did better. Its percentages were also far too high there (average 39% predicted
              against 6.9% actual), because half the training patients had heart disease.
            </p>
            <p>
              The main reason is blood pressure, the model’s strongest input, which barely separates sick from healthy
              people in NHANES, partly because most people already diagnosed take blood-pressure or cholesterol
              medication. So the scores are only meaningful for populations like the training data.
            </p>
          </div>
          <p className="mt-4 text-[0.95rem] text-stone-2">From src/external_validation.py.</p>
        </FramedPanel>

        <div className="grid gap-7 md:grid-cols-2">
          <FramedPanel corners={false} aria-labelledby="about-uncertainty">
            <h2 id="about-uncertainty" className="panel__title">
              Uncertainty ranges
            </h2>
            <div className="rule" />
            <div className="space-y-3 leading-relaxed text-stone">
              <p>
                Every prediction comes with a calibrated range from Venn–Abers predictors, a conformal prediction
                method: how often held-out patients with similar scores actually had heart disease.
              </p>
              <p>
                Calibrated on {uncertainty.calibration_patients.toLocaleString()} held-out patients and checked on
                another {uncertainty.evaluation_patients.toLocaleString()}, the ranges are usually narrow (median{' '}
                {uncertainty.range_width_points.median} points, 90% within {uncertainty.range_width_points.p90}) but
                widen to {uncertainty.range_width_points.max} points for unusual scores. They also show the model’s
                slight overconfidence at the very top of the scale.
              </p>
            </div>
          </FramedPanel>

          <FramedPanel corners={false} aria-labelledby="about-suggestions">
            <h2 id="about-suggestions" className="panel__title">
              Suggested changes
            </h2>
            <div className="rule" />
            <div className="space-y-3 leading-relaxed text-stone">
              <p>
                The Results page tries every combination of realistic, healthier changes (blood pressure, weight,
                smoking, alcohol, activity, cholesterol and glucose) and shows the best plan with one, two and three
                changes. Age, sex, height and medication never change.
              </p>
              <p>
                It also exposes a quirk of the clinic model: smokers in the Kaggle data have slightly lower recorded
                disease rates, so the model rarely credits quitting. The suggestions describe the model, not proven
                medical effects.
              </p>
            </div>
          </FramedPanel>
        </div>

        <FramedPanel corners={false} aria-labelledby="about-nhanes">
          <h2 id="about-nhanes" className="panel__title">
            US population model
          </h2>
          <div className="rule" />
          {nhanes ? (
            <>
              <p className="leading-relaxed text-stone">
                A second model trained on {nhanes.train.patients.toLocaleString()} US adults from NHANES{' '}
                {nhanes.train.cycles.join(', ')}, with blood-pressure and cholesterol medication as extra inputs, then
                tested on {nhanes.test.patients.toLocaleString()} adults from the later {nhanes.test_cycle} survey it
                never saw.
              </p>
              <div className="mt-5 overflow-x-auto">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>On the unseen {nhanes.test_cycle} survey</th>
                      <th>ROC-AUC</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="text-parchment">US population model</td>
                      <td>
                        {nhanes.test.roc_auc.toFixed(3)} (95% CI {nhanes.test.roc_auc_95ci[0].toFixed(3)}–
                        {nhanes.test.roc_auc_95ci[1].toFixed(3)})
                      </td>
                    </tr>
                    <tr>
                      <td className="text-parchment">Same model without medication inputs</td>
                      <td>{nhanes.ablation_without_medication.roc_auc.toFixed(3)}</td>
                    </tr>
                    <tr>
                      <td className="text-parchment">Ranking by age alone</td>
                      <td>{nhanes.test.roc_auc_age_alone.toFixed(3)}</td>
                    </tr>
                    <tr>
                      <td className="text-parchment">Logistic regression, same inputs</td>
                      <td>{nhanes.test.roc_auc_logistic.toFixed(3)}</td>
                    </tr>
                    <tr>
                      <td className="text-parchment">Ages 30–65: this model vs the clinic model</td>
                      <td>
                        {nhanes.comparison_age_30_65.us_population_model.toFixed(3)} vs{' '}
                        {nhanes.comparison_age_30_65.clinic_model.toFixed(3)}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p className="mt-4 leading-relaxed text-stone">
                Average predicted risk {pct(nhanes.test.mean_predicted_risk)} against {pct(nhanes.test.cvd_rate)}{' '}
                observed (calibration error {pct(nhanes.test.ece)}). Medication partly reflects treatment after a
                diagnosis, so this model recognises existing heart disease; it does not forecast future risk.
              </p>
            </>
          ) : (
            <p className="text-stone">Not trained yet. Run python src/train_nhanes_model.py to create it.</p>
          )}
        </FramedPanel>

        <FramedPanel corners={false} aria-labelledby="about-monitoring">
          <h2 id="about-monitoring" className="panel__title">
            Monitoring without patient data
          </h2>
          <div className="rule" />
          <p className="leading-relaxed text-stone">
            The server keeps only daily counts in coarse buckets: how many predictions, response times, the risk mix,
            and how recent inputs compare with each model’s training data. No inputs, exact values or IP addresses are
            stored, counts under 5 are hidden, and drift waits for 30 predictions.{' '}
            <Link to="/monitoring" className="text-gold-bright underline decoration-gold/50 underline-offset-4 hover:text-cream">
              See the live numbers
            </Link>
            .
          </p>
        </FramedPanel>

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
