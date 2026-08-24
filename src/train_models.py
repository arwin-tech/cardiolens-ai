import pandas as pd
import numpy as np
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class ModelBenchmark:
    def __init__(self, X_train, X_test, y_train, y_test, feature_names, random_state=42):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.random_state = random_state
        
        self.results = {}
        self.best_model_name = None
        self.best_pipeline = None
        self.best_metrics = None
        self.roc_curves = {}

    def _create_pipelines(self):
        pipelines = {}
        
        pipelines['Logistic Regression'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='lbfgs'
            ))
        ])
        
        pipelines['Random Forest'] = Pipeline([
            ('model', RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0
            ))
        ])
        
        pipelines['XGBoost'] = Pipeline([
            ('model', XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                verbosity=0,
                n_jobs=-1
            ))
        ])
        
        return pipelines

    def _evaluate_model(self, model_name, pipeline):
        print(f"\n[TRAIN] {model_name}...")
        
        pipeline.fit(self.X_train, self.y_train)
        
        y_pred = pipeline.predict(self.X_test)
        y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
        }
        
        cm = confusion_matrix(self.y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        metrics['roc_auc_value'] = auc(fpr, tpr)
        self.roc_curves[model_name] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': metrics['roc_auc_value']
        }
        
        print(f"    ✓ Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    ✓ Precision: {metrics['precision']:.4f}")
        print(f"    ✓ Recall:    {metrics['recall']:.4f}")
        print(f"    ✓ F1:        {metrics['f1']:.4f}")
        print(f"    ✓ ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"    ✓ Confusion Matrix:")
        print(f"        TP={cm[1,1]}, FP={cm[0,1]}, FN={cm[1,0]}, TN={cm[0,0]}")
        
        return metrics, pipeline, y_pred, y_pred_proba

    def benchmark(self):
        print("="*80)
        print("CardioLens AI - Stage 2: Model Benchmarking & Selection")
        print("="*80)
        
        print(f"\n[LOAD] Training/test data:")
        print(f"    X_train: {self.X_train.shape}")
        print(f"    X_test: {self.X_test.shape}")
        print(f"    y_train class distribution: {np.bincount(self.y_train)}")
        print(f"    y_test class distribution: {np.bincount(self.y_test)}")
        
        pipelines = self._create_pipelines()
        
        print(f"\n[BENCHMARK] Training and evaluating {len(pipelines)} models...")
        
        for model_name, pipeline in pipelines.items():
            metrics, fitted_pipeline, y_pred, y_pred_proba = self._evaluate_model(
                model_name, pipeline
            )
            self.results[model_name] = {
                'metrics': metrics,
                'pipeline': fitted_pipeline,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        
        print(f"\n[COMPARE] Model Comparison Table:")
        print("="*80)
        
        comparison_df = pd.DataFrame({
            model_name: results['metrics']
            for model_name, results in self.results.items()
        }).T
        
        display_df = comparison_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].copy()
        print(display_df.to_string())
        print("="*80)
        
        self.best_model_name = max(
            self.results.keys(),
            key=lambda x: self.results[x]['metrics']['roc_auc']
        )
        best_result = self.results[self.best_model_name]
        self.best_metrics = best_result['metrics']
        self.best_pipeline = best_result['pipeline']
        
        print(f"\n[SELECT] Best Model: {self.best_model_name}")
        print(f"         ROC-AUC: {self.best_metrics['roc_auc']:.4f}")
        print(f"         Accuracy: {self.best_metrics['accuracy']:.4f}")
        print(f"         F1 Score: {self.best_metrics['f1']:.4f}")
        
        return {
            'best_model_name': self.best_model_name,
            'best_metrics': self.best_metrics,
            'best_pipeline': self.best_pipeline,
            'all_results': self.results,
            'roc_curves': self.roc_curves
        }

    def save_artifacts(self, output_dir='models'):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n[SAVE] Saving artifacts to {output_dir}...")
        
        pipeline_path = output_dir / 'model_pipeline.pkl'
        with open(pipeline_path, 'wb') as f:
            pickle.dump(self.best_pipeline, f)
        print(f"    ✓ Pipeline saved to: {pipeline_path}")
        
        # Retrieve model class safely whether scaler exists or not
        if 'model' in self.best_pipeline.named_steps:
            model_obj = self.best_pipeline.named_steps['model']
        else:
            model_obj = self.best_pipeline.steps[-1][1]

        metadata = {
            'model_type': self.best_model_name,
            'model_class': str(model_obj.__class__),
            'test_metrics': {
                'accuracy': float(self.best_metrics['accuracy']),
                'precision': float(self.best_metrics['precision']),
                'recall': float(self.best_metrics['recall']),
                'f1': float(self.best_metrics['f1']),
                'roc_auc': float(self.best_metrics['roc_auc']),
            },
            'confusion_matrix': self.best_metrics['confusion_matrix'],
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'n_train_samples': len(self.X_train),
            'n_test_samples': len(self.X_test),
            'training_date': datetime.now().isoformat(),
            'dataset': 'Kaggle Cardiovascular Disease',
            'random_state': self.random_state,
            'note': 'This model was selected based on highest test ROC-AUC score'
        }
        
        metadata_path = output_dir / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"    ✓ Metadata saved to: {metadata_path}")
        
        self._save_confusion_matrix_plot(output_dir)
        self._save_roc_curve_plot(output_dir)
        
        print(f"    ✅ All artifacts saved!")

    def _save_confusion_matrix_plot(self, output_dir):
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = np.array(self.best_metrics['confusion_matrix'])
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'},
            xticklabels=['Predicted Healthy', 'Predicted Disease'],
            yticklabels=['Actual Healthy', 'Actual Disease']
        )
        ax.set_title(f'Confusion Matrix - {self.best_model_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        plot_path = output_dir / 'confusion_matrix.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Confusion matrix plot saved to: {plot_path}")

    def _save_roc_curve_plot(self, output_dir):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC=0.5)')
        colors = {'Logistic Regression': '#1f77b4', 'Random Forest': '#ff7f0e', 'XGBoost': '#2ca02c'}
        
        for model_name, roc_data in self.roc_curves.items():
            fpr = roc_data['fpr']
            tpr = roc_data['tpr']
            auc_score = roc_data['auc']
            
            linewidth = 3 if model_name == self.best_model_name else 2
            linestyle = '-' if model_name == self.best_model_name else '--'
            label = f"{model_name} (AUC={auc_score:.4f})"
            if model_name == self.best_model_name:
                label += " ← BEST"
            
            ax.plot(
                fpr, tpr,
                color=colors.get(model_name, '#000000'),
                lw=linewidth,
                linestyle=linestyle,
                label=label
            )
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve Comparison - All Models', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / 'roc_curve_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    ✓ ROC curve comparison saved to: {plot_path}")


def load_stage1_data():
    print("\n[LOAD] Loading Stage 1 output...")
    from src.data_loader import main as stage1_main
    stage1_output = stage1_main()
    
    if stage1_output is None:
        raise RuntimeError("Stage 1 failed. Cannot proceed with Stage 2.")
    
    return (
        stage1_output['X_train'],
        stage1_output['X_test'],
        stage1_output['y_train'],
        stage1_output['y_test'],
        stage1_output['feature_names']
    )


def main():
    X_train, X_test, y_train, y_test, feature_names = load_stage1_data()
    benchmarker = ModelBenchmark(X_train, X_test, y_train, y_test, feature_names, random_state=42)
    results = benchmarker.benchmark()
    benchmarker.save_artifacts('models')
    
    print("\n" + "="*80)
    print("✅ Stage 2 Complete!")
    print("="*80)

if __name__ == "__main__":
    main()