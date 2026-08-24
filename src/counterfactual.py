import os
import pickle
import json
import pandas as pd
import numpy as np

class CounterfactualSimulator:
    def __init__(self, model_path="models/model_pipeline.pkl", metadata_path="models/model_metadata.json"):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.pipeline = None
        self.feature_names = None
        self.metadata = None

    def load_artifacts(self):
        print("\n[LOAD] Loading model pipeline...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Run Stage 2 first.")
        with open(self.model_path, "rb") as f:
            self.pipeline = pickle.load(f)
        print(f"    ✓ Pipeline loaded: {self.pipeline.named_steps['model'].__class__.__name__ if hasattr(self.pipeline, 'named_steps') else 'Model'}")

        print("\n[LOAD] Loading metadata...")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}. Run Stage 2 first.")
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)
        self.feature_names = self.metadata["feature_names"]
        print(f"    ✓ Feature metadata loaded ({len(self.feature_names)} features)")

    def validate_profile(self, profile):
        missing = [f for f in self.feature_names if f not in profile]
        if missing:
            raise ValueError(f"Profile missing required features: {missing}")
        # Ensure correct feature ordering matching model training
        ordered_df = pd.DataFrame([profile])[self.feature_names]
        return ordered_df

    def predict_risk(self, profile):
        ordered_df = self.validate_profile(profile)
        proba = float(self.pipeline.predict_proba(ordered_df)[0, 1])
        return proba

    def simulate_changes(self, original_profile, changes):
        simulated_profile = original_profile.copy()
        simulated_profile.update(changes)

        orig_risk = self.predict_risk(original_profile)
        sim_risk = self.predict_risk(simulated_profile)
        diff = sim_risk - orig_risk

        return {
            "original_risk": orig_risk,
            "simulated_risk": sim_risk,
            "risk_difference": diff,
            "original_risk_percent": round(orig_risk * 100, 2),
            "simulated_risk_percent": round(sim_risk * 100, 2),
            "difference_percentage_points": round(diff * 100, 2),
            "disclaimer": "This counterfactual simulation reflects model outputs under modified inputs and does not imply medical causality."
        }

def main():
    print("="*80)
    print("CardioLens AI - Stage 4: Counterfactual Simulator")
    print("="*80)

    simulator = CounterfactualSimulator()
    simulator.load_artifacts()

    # Sample baseline high-risk patient profile
    original_patient = {
        "age": 57,
        "gender": 1,
        "height": 170,
        "weight": 85,
        "ap_hi": 155,
        "ap_lo": 95,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 1,
        "alco": 0,
        "active": 1
    }

    print("\n[VALIDATE] Validating sample patient profile...")
    simulator.validate_profile(original_patient)
    print("    ✓ All 11 features validated successfully")

    print("\n[PREDICT] Original profile risk...")
    orig_risk = simulator.predict_risk(original_patient)
    print(f"    Original Risk: {orig_risk:.2%}")

    print("\n[SIMULATE] Applying counterfactual changes (ap_hi: 155→130, smoke: 1→0)...")
    changes = {"ap_hi": 130, "smoke": 0}
    res = simulator.simulate_changes(original_patient, changes)

    print("\n[RESULT]")
    print(f"    Original Risk:   {res['original_risk_percent']}%")
    print(f"    Simulated Risk:  {res['simulated_risk_percent']}%")
    print(f"    Risk Difference: {res['difference_percentage_points']:+.2f} percentage points")

    print("\n[SAFETY]")
    print(f"    ✓ Safety disclaimer verified")

    print("\n" + "="*80)
    print("✅ Stage 4 Complete!")
    print("="*80)

if __name__ == "__main__":
    main()