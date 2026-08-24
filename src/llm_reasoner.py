import json
from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod
from pathlib import Path


class LLMReasoner(ABC):
    """
    Abstract base class for LLM reasoning. Subclass this to integrate any LLM.
    """
    def __init__(self, model_name: str, model_config: Optional[Dict] = None):
        self.model_name = model_name
        self.model_config = model_config or {}

    @abstractmethod
    def generate_risk_assessment(self, risk_prob: float, feature_values: Dict) -> str:
        pass

    @abstractmethod
    def generate_shap_explanation(self, top_factors: List[Dict], base_value: float) -> str:
        pass

    @abstractmethod
    def generate_counterfactual_explanation(
        self,
        original_risk: float,
        simulated_risk: float,
        changes: Dict[str, Union[float, int]]
    ) -> str:
        pass

    def get_safety_disclaimer(self) -> str:
        return (
            "\n⚠️ RESEARCH PROTOTYPE DISCLAIMER:\n"
            "CardioLens AI is an educational/research prototype, not a medical diagnostic tool.\n"
            "This explanation describes model behavior based on input features.\n"
            "It does NOT establish medical causality or provide medical diagnosis.\n"
            "It does NOT recommend treatment or lifestyle changes.\n"
            "Consult qualified healthcare professionals for medical advice."
        )


class DeterministicReasoner(LLMReasoner):
    """
    Deterministic placeholder reasoner (no external API calls).
    """
    def generate_risk_assessment(self, risk_prob: float, feature_values: Dict) -> str:
        if risk_prob < 0.3:
            assessment = "The model predicts a low cardiovascular risk based on the input profile."
        elif risk_prob < 0.6:
            assessment = "The model identifies moderate cardiovascular risk factors in this profile."
        elif risk_prob < 0.8:
            assessment = "The model identifies elevated cardiovascular risk based on the clinical profile."
        else:
            assessment = "The model predicts high cardiovascular risk probability for this patient profile."

        key_factors = []
        age = feature_values.get('age', 0)
        if age > 50:
            key_factors.append(f"age ({age} years) is a significant factor")

        ap_hi = feature_values.get('ap_hi', 0)
        if ap_hi > 140:
            key_factors.append(f"systolic blood pressure ({ap_hi} mmHg) is notably elevated")

        smoke = feature_values.get('smoke', 0)
        if smoke == 1:
            key_factors.append("smoking status is a contributing factor")

        cholesterol = feature_values.get('cholesterol', 1)
        if cholesterol > 1:
            key_factors.append(f"cholesterol level is above normal (level {cholesterol})")

        gluc = feature_values.get('gluc', 1)
        if gluc > 1:
            key_factors.append(f"glucose level is above normal (level {gluc})")

        factors_text = ""
        if key_factors:
            factors_text = f"\n\nFactors contributing to this assessment:\n" + "\n".join(f"• {f}" for f in key_factors)

        return f"{assessment}{factors_text}"

    def generate_shap_explanation(self, top_factors: List[Dict], base_value: float) -> str:
        if not top_factors:
            return "No significant factors identified in the analysis."

        explanation = (
            f"The model analysis shows the following feature contributions to the prediction:\n\n"
            f"Base model expectation: {base_value:.1%} risk\n\n"
            "Top contributing factors:"
        )

        for i, factor in enumerate(top_factors[:5], 1):
            feature = factor.get('feature', 'unknown')
            value = factor.get('value', 0)
            contribution = factor.get('contribution', 0)

            direction = "increases" if contribution > 0 else "decreases"
            symbol = "↑" if contribution > 0 else "↓"

            explanation += (
                f"\n{i}. {feature} (value: {value}): {symbol} {direction} risk by {abs(contribution):.1%}"
            )

        return explanation

    def generate_counterfactual_explanation(
        self,
        original_risk: float,
        simulated_risk: float,
        changes: Dict[str, Union[float, int]]
    ) -> str:
        risk_change = simulated_risk - original_risk
        changes_text = "\n".join(f"• {k}: {v}" for k, v in changes.items())

        magnitude = "substantially" if abs(risk_change) > 0.1 else "slightly"
        direction = "lower" if risk_change < 0 else "higher"

        explanation = (
            f"Counterfactual Analysis Results:\n\n"
            f"Modified Features:\n{changes_text}\n\n"
            f"Original risk prediction: {original_risk:.1%}\n"
            f"Simulated risk prediction: {simulated_risk:.1%}\n"
            f"Model-predicted change: {risk_change:+.1%}\n\n"
            f"Interpretation:\n"
            f"Under the modified profile, the model predicts a {magnitude} {direction} risk "
            f"(difference of {abs(risk_change):.1%} percentage points) compared to the original profile.\n\n"
            f"IMPORTANT: This represents model behavior under hypothetical inputs. "
            f"It does NOT establish causality or predict real-world outcomes."
        )

        return explanation


class StructuredReasoner(LLMReasoner):
    """
    Structured output reasoner - returns JSON instead of text.
    """
    def generate_risk_assessment(self, risk_prob: float, feature_values: Dict) -> str:
        result = {
            "type": "risk_assessment",
            "risk_probability": round(risk_prob, 4),
            "risk_percentage": round(risk_prob * 100, 1),
            "risk_category": "high" if risk_prob >= 0.8 else "elevated" if risk_prob >= 0.6 else "moderate" if risk_prob >= 0.3 else "low",
            "summary": f"Model predicts high cardiovascular risk ({risk_prob:.1%}) for this patient profile.",
            "model": "XGBoost",
            "disclaimer": "..."
        }
        return json.dumps(result, indent=2)

    def generate_shap_explanation(self, top_factors: List[Dict], base_value: float) -> str:
        result = {
            "type": "shap_explanation",
            "base_value": round(base_value, 4),
            "top_factors": top_factors[:5],
            "summary": "SHAP analysis shows the relative contribution of each feature to the model's prediction.",
            "disclaimer": self.get_safety_disclaimer()
        }
        return json.dumps(result, indent=2)

    def generate_counterfactual_explanation(
        self,
        original_risk: float,
        simulated_risk: float,
        changes: Dict[str, Union[float, int]]
    ) -> str:
        result = {
            "type": "counterfactual_explanation",
            "original_risk": round(original_risk, 4),
            "simulated_risk": round(simulated_risk, 4),
            "risk_change": round(simulated_risk - original_risk, 4),
            "modified_features": changes,
            "summary": f"Under the modified profile, the model predicts risk of {simulated_risk:.1%} compared to {original_risk:.1%} in the original profile.",
            "disclaimer": self.get_safety_disclaimer()
        }
        return json.dumps(result, indent=2)


def main():
    print("="*80)
    print("CardioLens AI - Stage 5: LLM Reasoning Layer")
    print("="*80)

    patient_features = {
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
    risk_prob = 0.857

    print("\n[TEST 1] DeterministicReasoner")
    print("─" * 76)

    reasoner = DeterministicReasoner(model_name="deterministic-v1")

    print("\n[Risk Assessment]")
    print(reasoner.generate_risk_assessment(risk_prob, patient_features))

    top_factors = [
        {"feature": "age", "value": 57, "contribution": 0.15},
        {"feature": "ap_hi", "value": 155, "contribution": 0.20},
        {"feature": "smoke", "value": 1, "contribution": 0.12},
        {"feature": "cholesterol", "value": 2, "contribution": 0.08},
    ]

    print("\n[SHAP Explanation]")
    print(reasoner.generate_shap_explanation(top_factors, base_value=0.50))

    print("\n[Counterfactual Explanation]")
    print(reasoner.generate_counterfactual_explanation(
        original_risk=0.857,
        simulated_risk=0.6664,
        changes={"ap_hi": 130, "smoke": 0}
    ))

    print(reasoner.get_safety_disclaimer())

    print("\n" + "="*80)
    print("[TEST 2] StructuredReasoner (JSON output)")
    print("─" * 76)

    json_reasoner = StructuredReasoner(model_name="structured-json")
    print("\n[Risk Assessment (JSON)]")
    print(json_reasoner.generate_risk_assessment(risk_prob, patient_features))

    print("\n" + "="*80)
    print("[INFO] Stage 5 is modular. To use a different LLM:")
    print("─" * 76)
    print("""
class OpenAIReasoner(LLMReasoner):
    def __init__(self, api_key, model="gpt-4"):
        super().__init__(model_name=model)
        self.client = OpenAI(api_key=api_key)
    
    def generate_risk_assessment(self, risk_prob, feature_values):
        prompt = f"Explain this cardiovascular risk prediction: {risk_prob:.1%}"
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

# Usage:
reasoner = OpenAIReasoner(api_key="sk-...")
explanation = reasoner.generate_risk_assessment(0.857, patient_features)
""")

    print("="*80)
    print("✅ Stage 5 Complete!")
    print("="*80)

if __name__ == "__main__":
    main()