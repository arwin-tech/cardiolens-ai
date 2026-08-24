import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.data_loader import main as stage1_main

def generate_shap_explanations():
    print("="*80)
    print("CardioLens AI - Stage 3: SHAP Explainability")
    print("="*80)

    pipeline_path = "models/model_pipeline.pkl"
    metadata_path = "models/model_metadata.json"

    print("\n[VALIDATE] Checking Stage 2 artifacts...")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Pipeline file not found at {pipeline_path}. Run Stage 2 first.")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Run Stage 2 first.")
    
    print(f"    ✓ Pipeline file found: {pipeline_path}")
    print(f"    ✓ Metadata file found: {metadata_path}")

    print("\n[LOAD] Loading Stage 2 artifacts and test data...")
    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    stage1_output = stage1_main()
    X_test = stage1_output['X_test']
    feature_names = metadata['feature_names']

    print("\n[EXTRACT] Extracting estimator from pipeline...")
    if hasattr(pipeline, 'named_steps') and 'model' in pipeline.named_steps:
        model = pipeline.named_steps['model']
    else:
        model = pipeline.steps[-1][1]
    
    print(f"    ✓ Extracted model: {model.__class__.__name__}")

    if hasattr(pipeline, 'named_steps') and 'scaler' in pipeline.named_steps:
        X_test_transformed = pipeline.named_steps['scaler'].transform(X_test)
        X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
    else:
        X_test_df = pd.DataFrame(X_test, columns=feature_names)

    print("\n[EXPLAINER] Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    print("\n[SHAP] Computing SHAP values (sample size: 1000)...")
    np.random.seed(42)
    sample_size = min(1000, len(X_test_df))
    sample_indices = np.random.choice(len(X_test_df), size=sample_size, replace=False)
    shap_sample = X_test_df.iloc[sample_indices]

    shap_values = explainer(shap_sample)
    print(f"    ✓ SHAP values computed: shape {shap_values.values.shape}")

    os.makedirs("explainability", exist_ok=True)

    print("\n[PLOTS] Generating global SHAP visualizations...")
    
    # 1. Feature Importance Bar Chart
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.title("SHAP Feature Importance (Bar Plot)", fontsize=14, pad=15)
    plt.tight_layout()
    bar_path = "explainability/shap_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"        ✓ Saved: {bar_path}")

    # 2. Beeswarm Summary Plot
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Value Distribution (Beeswarm Plot)", fontsize=14, pad=15)
    plt.tight_layout()
    summary_path = "explainability/shap_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"        ✓ Saved: {summary_path}")

    print("\n[LOCAL] Generating patient-level explanations...")
    preds_proba = pipeline.predict_proba(shap_sample)[:, 1]
    
    local_explanations = []
    for idx in range(min(5, len(shap_sample))):
        patient_shap = shap_values.values[idx]
        patient_features = shap_sample.iloc[idx].to_dict()
        
        sorted_feat_indices = np.argsort(np.abs(patient_shap))[::-1]
        top_factors = []
        for f_idx in sorted_feat_indices[:5]:
            top_factors.append({
                "feature": feature_names[f_idx],
                "value": float(patient_features[feature_names[f_idx]]),
                "shap_value": float(patient_shap[f_idx])
            })

        local_explanations.append({
            "patient_id": f"patient_{sample_indices[idx]}",
            "predicted_risk": float(preds_proba[idx]),
            "top_factors": top_factors,
            "disclaimer": "This explanation describes model behavior only, not medical causality."
        })

    local_path = "explainability/local_explanations.json"
    with open(local_path, "w") as f:
        json.dump(local_explanations, f, indent=2)
    print(f"    ✓ Saved local explanations: {local_path}")

    shap_meta = {
        "model_type": model.__class__.__name__,
        "sample_size": sample_size,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "disclaimer": "SHAP explanations reflect statistical association within the trained model and do not constitute clinical guidance."
    }
    meta_path = "explainability/shap_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(shap_meta, f, indent=2)
    print(f"\n[METADATA] Saved SHAP metadata to: {meta_path}")

    print("\n" + "="*80)
    print("✅ Stage 3 Complete!")
    print("="*80)

if __name__ == "__main__":
    generate_shap_explanations()