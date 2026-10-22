import { HeroBackdrop } from '../components/Backdrop';
import HeartECG from '../components/HeartECG';
import ArchCard from '../components/ArchCard';
import { OrnateLink } from '../components/ornaments';
import engineArt from '../assets/card-engine.webp';
import shapArt from '../assets/card-shap.webp';
import counterfactualArt from '../assets/card-counterfactual.webp';

export default function Home() {
  return (
    <section className="hero">
      <div className="hero__frame">
        <HeroBackdrop />
        <HeartECG className="hero__heart" />

        <div className="hero__copy">
          <h1 className="hero__title">
            Explainable AI for <span className="hero__title-accent">Cardiovascular</span>
            <br className="hidden sm:inline" /> Health Assessment
          </h1>
          <p className="hero__lede">
            CardioLens AI combines robust gradient-boosted ML pipelines (XGBoost) with SHAP explainability to evaluate
            risk scores and counterfactual intervention scenarios.
          </p>
          <div className="hero__actions">
            <OrnateLink to="/analyze" tone="crimson">
              Launch Clinical Analysis
            </OrnateLink>
            <OrnateLink to="/about" tone="umber">
              View Model Architecture
            </OrnateLink>
          </div>
        </div>

        <div className="hero__cards">
          <ArchCard art={engineArt} title="Gradient Boosted Engine">
            Trained on over 70,000 patient records to provide accurate baseline cardiovascular risk probabilities.
          </ArchCard>
          <ArchCard art={shapArt} title="Local SHAP Explainability">
            Inspect individual feature contribution rankings to see exactly which risk factors drove the output score.
          </ArchCard>
          <ArchCard art={counterfactualArt} title="Counterfactual Analysis">
            Simulate lifestyle or blood pressure changes dynamically to estimate risk delta before clinical
            intervention.
          </ArchCard>
        </div>
      </div>
    </section>
  );
}
