import { ChangeEvent, FormEvent, ReactNode, useState } from 'react';
import { PatientInput } from '../types/api';
import { FramedPanel, OrnateButton } from './ornaments';
import { EXAMPLES } from '../lib/examples';


interface Props {
  onSubmit: (data: PatientInput) => void;
  loading: boolean;
  waking?: boolean;
}

function Field({ id, label, hint, wide, children }: { id: string; label: string; hint?: string; wide?: boolean; children: ReactNode }) {
  return (
    <div className={wide ? 'md:col-span-2' : undefined}>
      <label htmlFor={id} className="field-label">
        {label} {hint && <span className="field-hint">({hint})</span>}
      </label>
      {children}
    </div>
  );
}

export default function PatientForm({ onSubmit, loading, waking = false }: Props) {
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
  const [formError, setFormError] = useState<string | null>(null);

  const handleChange = (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: Number(value) }));
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (formData.ap_lo >= formData.ap_hi) {
      setFormError('Diastolic pressure must be lower than systolic pressure.');
      return;
    }
    setFormError(null);
    onSubmit(formData);
  };

  const number = (name: keyof PatientInput, min: number, max: number, step = 1) => (
    <input
      id={`f-${name}`}
      className="field"
      type="number"
      name={name}
      min={min}
      max={max}
      step={step}
      value={formData[name]}
      onChange={handleChange}
      required
    />
  );

  const select = (name: keyof PatientInput, options: [number, string][]) => (
    <select id={`f-${name}`} className="field" name={name} value={formData[name]} onChange={handleChange}>
      {options.map(([value, text]) => (
        <option key={value} value={value}>
          {text}
        </option>
      ))}
    </select>
  );

  const levels: [number, string][] = [
    [1, 'Normal'],
    [2, 'Above normal'],
    [3, 'Well above normal'],
  ];
  const yesNo: [number, string][] = [
    [0, 'No'],
    [1, 'Yes'],
  ];

  return (
    <FramedPanel aria-labelledby="patient-details-title">
      <form onSubmit={handleSubmit}>
        <h2 id="patient-details-title" className="panel__title">
          Patient details
        </h2>
        <div className="rule" />

        <div className="mb-8 flex flex-wrap items-center gap-2.5">
          <span className="mr-1 text-stone">Try an example:</span>
          {EXAMPLES.map((ex) => (
            <button
              key={ex.label}
              type="button"
              className="chip"
              disabled={loading}
              onClick={() => {
                setFormData(ex.data);
                setFormError(null);
                onSubmit(ex.data);
              }}
            >
              <span className="chip__dot" style={{ background: ex.color }} aria-hidden="true" />
              {ex.label}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 gap-x-6 gap-y-5 md:grid-cols-2">
          <Field id="f-age" label="Age" hint="years">{number('age', 18, 100)}</Field>
          <Field id="f-gender" label="Gender">{select('gender', [[1, 'Female'], [2, 'Male']])}</Field>
          <Field id="f-height" label="Height" hint="cm">{number('height', 100, 250)}</Field>
          <Field id="f-weight" label="Weight" hint="kg">{number('weight', 30, 300, 0.1)}</Field>
          <Field id="f-ap_hi" label="Systolic blood pressure" hint="mmHg">{number('ap_hi', 60, 250)}</Field>
          <Field id="f-ap_lo" label="Diastolic blood pressure" hint="mmHg">{number('ap_lo', 30, 180)}</Field>
          <Field id="f-cholesterol" label="Cholesterol">{select('cholesterol', levels)}</Field>
          <Field id="f-gluc" label="Glucose">{select('gluc', levels)}</Field>
          <Field id="f-smoke" label="Smoker">{select('smoke', yesNo)}</Field>
          <Field id="f-alco" label="Regular alcohol intake">{select('alco', yesNo)}</Field>
          <Field id="f-active" label="Physical activity" wide>
            {select('active', [[1, 'Active lifestyle'], [0, 'Sedentary lifestyle']])}
          </Field>
        </div>

        {formError && (
          <p className="notice notice--error mt-6" role="alert">
            {formError}
          </p>
        )}

        <div className="mt-9">
          <OrnateButton type="submit" tone="crimson" block disabled={loading}>
            {loading ? (waking ? 'Waking up the model (up to a minute)…' : 'Calculating risk…') : 'Evaluate risk score'}
          </OrnateButton>
        </div>
      </form>
    </FramedPanel>
  );
}
