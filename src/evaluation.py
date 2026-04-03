"""
Evaluation Module
=================
Comprehensive evaluation and visualization of model predictions and trajectories.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

# Use non-interactive backend for compatibility
matplotlib.use('Agg')


class TrajectoryEvaluator:
    """Evaluate and visualize trajectory model results."""
    
    def __init__(self, results_dir: str = 'results'):
        """
        Initialize evaluator.
        
        Parameters:
            results_dir: Directory to save plots
        """
        self.results_dir = results_dir
        sns.set_style("whitegrid")
    
    def plot_roc_curves(self, model_results: Dict, output_path: str = None):
        """
        Plot ROC curves for all models.
        
        Parameters:
            model_results: Dictionary with model metrics including roc_curve data
            output_path: Path to save plot
        """
        print("\n" + "=" * 60)
        print("EVALUATION & VISUALIZATION")
        print("=" * 60)
        print("Plotting ROC curves...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, result in model_results.items():
            if 'roc_curve' in result['metrics']:
                fpr, tpr, _ = result['metrics']['roc_curve']
                auc_score = result['metrics']['auc']
                
                ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})',
                       linewidth=2)
        
        # Random classifier baseline
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)',
               linewidth=1.5)
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - ICU Trajectory Models', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"? ROC curve saved to {output_path}")
        
        return fig
    
    def plot_confusion_matrices(self, model_results: Dict, output_path: str = None):
        """
        Plot confusion matrices for all models.
        
        Parameters:
            model_results: Dictionary with model metrics
            output_path: Path to save plot
        """
        print("Plotting confusion matrices...")
        
        num_models = len(model_results)
        cols = min(num_models, 3)
        rows = (num_models + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes = np.array(axes).flatten()
        
        for idx, (model_name, result) in enumerate(model_results.items()):
            cm = result['metrics']['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar=False, annot_kws={'size': 14})
            
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=10)
            axes[idx].set_xlabel('Predicted Label', fontsize=10)
            axes[idx].set_xticklabels(['Stable', 'Deteriorating'])
            axes[idx].set_yticklabels(['Stable', 'Deteriorating'])
        
        # Hide extra subplots
        for idx in range(num_models, len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"? Confusion matrices saved to {output_path}")
        
        return fig
    
    def plot_trajectory_examples(self, df: pd.DataFrame, predictions: Dict[str, np.ndarray],
                                sample_patients: int = 3, output_path: str = None):
        """
        Plot actual trajectories with predicted deterioration labels over time.
        
        Parameters:
            df: DataFrame with vital signs and true labels
            predictions: Dictionary mapping model names to prediction arrays
            sample_patients: Number of patients to plot
            output_path: Path to save plot
        """
        print("Plotting trajectory examples...")
        
        unique_patients = df['patient_id'].unique()[:sample_patients]
        
        fig, axes = plt.subplots(len(unique_patients), 2, figsize=(15, 4*len(unique_patients)))
        if len(unique_patients) == 1:
            axes = axes.reshape(1, -1)
        
        for ax_idx, patient_id in enumerate(unique_patients):
            patient_data = df[df['patient_id'] == patient_id].sort_values('timestamp')
            
            # Plot vital signs
            ax = axes[ax_idx, 0]
            vitals = ['heart_rate', 'systolic_bp', 'respiratory_rate']
            
            for vital in vitals:
                ax.plot(range(len(patient_data)), patient_data[vital].values,
                       marker='o', label=vital, alpha=0.7)
            
            ax.set_title(f'Patient {patient_id} - Vital Signs Trajectories', fontweight='bold')
            ax.set_xlabel('Time Index')
            ax.set_ylabel('Normalized Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot labels and predictions
            ax = axes[ax_idx, 1]
            
            time_idx = np.arange(len(patient_data))
            
            # True label
            ax.fill_between(time_idx, 0, patient_data['label'].values, alpha=0.3,
                           label='True: Deteriorating', step='mid', color='red')
            
            # Predicted label (if available)
            if 'predicted_deterioration' in patient_data.columns:
                ax.fill_between(time_idx, 0, patient_data['predicted_deterioration'].values,
                               alpha=0.3, label='Predicted: Deteriorating', 
                               step='mid', color='orange')
            
            ax.set_title(f'Patient {patient_id} - Deterioration Flags', fontweight='bold')
            ax.set_xlabel('Time Index')
            ax.set_ylabel('Deterioration Flag')
            ax.set_ylim([0, 1.2])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Example Patient Trajectories', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"? Trajectory examples saved to {output_path}")
        
        return fig
    
    def plot_training_history(self, train_losses: List[float], val_losses: List[float],
                             output_path: str = None):
        """
        Plot LSTM training history.
        
        Parameters:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
            output_path: Path to save plot
        """
        print("Plotting training history...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = np.arange(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'o-', label='Training Loss', linewidth=2, markersize=4)
        ax.plot(epochs, val_losses, 's-', label='Validation Loss', linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (BCEWithLogitsLoss)', fontsize=12)
        ax.set_title('LSTM Training History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"? Training history saved to {output_path}")
        
        return fig
    
    def plot_feature_importance(self, feature_importance_dict: Dict[str, pd.DataFrame],
                               top_n: int = 10, output_path: str = None):
        """
        Plot feature importance from ML models.
        
        Parameters:
            feature_importance_dict: Dictionary mapping model names to importance DataFrames
            top_n: Number of top features to display
            output_path: Path to save plot
        """
        print("Plotting feature importance...")
        
        num_models = len(feature_importance_dict)
        fig, axes = plt.subplots(1, num_models, figsize=(8*num_models, 6))
        
        if num_models == 1:
            axes = [axes]
        
        for ax_idx, (model_name, importance_df) in enumerate(feature_importance_dict.items()):
            top_features = importance_df.head(top_n)
            
            axes[ax_idx].barh(range(len(top_features)), top_features['importance'].values,
                            color='steelblue')
            axes[ax_idx].set_yticks(range(len(top_features)))
            axes[ax_idx].set_yticklabels(top_features['feature'].values)
            axes[ax_idx].set_xlabel('Importance Score', fontsize=11)
            axes[ax_idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[ax_idx].invert_yaxis()
            axes[ax_idx].grid(True, alpha=0.3, axis='x')
        
        fig.suptitle(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"? Feature importance plot saved to {output_path}")
        
        return fig
    
    def generate_summary_report(self, model_results: Dict, trajectory_metrics: Dict,
                               output_path: str = None) -> str:
        """
        Generate text summary report of all results.
        
        Parameters:
            model_results: Dictionary with ML and DL model results
            trajectory_metrics: Dictionary with trajectory detection metrics
            output_path: Path to save report
            
        Returns:
            Report string
        """
        print("Generating summary report...")
        
        report = []
        report.append("=" * 70)
        report.append("ICU TRAJECTORY DETERIORATION DETECTION - COMPREHENSIVE REPORT")
        report.append("=" * 70)
        
        report.append("\n1. MODEL PERFORMANCE COMPARISON")
        report.append("-" * 70)
        
        for model_name, result in model_results.items():
            metrics = result['metrics']
            report.append(f"\n{model_name}:")
            report.append(f"  Accuracy:  {metrics['accuracy']:.4f}")
            report.append(f"  Precision: {metrics['precision']:.4f}")
            report.append(f"  Recall:    {metrics['recall']:.4f}")
            report.append(f"  F1-Score:  {metrics['f1']:.4f}")
            report.append(f"  AUC-ROC:   {metrics['auc']:.4f}")
        
        report.append("\n\n2. TRAJECTORY-BASED DETECTION")
        report.append("-" * 70)
        
        for metric, value in trajectory_metrics.items():
            if not isinstance(value, np.ndarray):
                report.append(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")
        
        report.append("\n\n3. KEY FINDINGS")
        report.append("-" * 70)

        best_model = max(model_results.items(),
                         key=lambda x: x[1]['metrics']['f1'])
        best_auprc = max(model_results.items(),
                         key=lambda x: x[1]['metrics'].get('auprc', 0))
        report.append(f"Best F1:    {best_model[0]}  (F1={best_model[1]['metrics']['f1']:.4f})")
        report.append(f"Best AUPRC: {best_auprc[0]}  (AUPRC={best_auprc[1]['metrics'].get('auprc', 0):.4f})")

        report.append("\nNote: AUPRC is the primary metric for imbalanced ICU data.")
        report.append("AUC-ROC can be misleading at 2-3% positive rate.")
        report.append("\nRecommendations:")
        report.append("- XGBoost/LightGBM: best for tabular trajectory features")
        report.append("- LSTM: best for raw temporal sequence patterns")
        report.append("- Trajectory logic: most interpretable, clinically explainable")
        report.append("- Ensemble of all three is the IEEE novel contribution")

        report.append("\n" + "=" * 70)

        report_str = "\n".join(report)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_str)
            print(f"Report saved to {output_path}")

        return report_str

    def plot_shap_summary(self, model, X_test: np.ndarray, feature_names: list,
                          model_name: str = 'XGBoost', output_path: str = None):
        """
        Generate SHAP summary plot showing which features drive predictions.
        Required for IEEE medical ML papers — explains model decisions.
        Only works with tree-based models (RF, XGBoost, LightGBM).
        """
        try:
            import shap
        except ImportError:
            print("shap not installed — skipping SHAP plot")
            return

        print(f"Generating SHAP summary for {model_name}...")

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # For binary classification some models return list[neg, pos]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                          show=False, plot_type='bar')
        plt.title(f"SHAP Feature Importance - {model_name}", fontsize=14)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"SHAP plot saved to {output_path}")
        plt.close()

    def plot_precision_recall_curves(self, model_results: Dict,
                                     output_path: str = None):
        """
        Plot Precision-Recall curves for all models.
        More informative than ROC for imbalanced data (2-3% positive rate).
        """
        ax = plt.figure(figsize=(10, 8)).add_subplot(111)

        for model_name, result in model_results.items():
            metrics = result['metrics']
            if 'pr_curve' in metrics:
                prec, rec, _ = metrics['pr_curve']
                auprc = metrics.get('auprc', 0)
                ax.plot(rec, prec, lw=2, label=f"{model_name} (AUPRC={auprc:.3f})")
            elif 'y_test' in metrics and 'y_pred_proba' in metrics:
                from sklearn.metrics import precision_recall_curve, average_precision_score
                prec, rec, _ = precision_recall_curve(
                    metrics['y_test'], metrics['y_pred_proba'])
                auprc = average_precision_score(
                    metrics['y_test'], metrics['y_pred_proba'])
                ax.plot(rec, prec, lw=2, label=f"{model_name} (AUPRC={auprc:.3f})")

        ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curves\n(Primary metric for imbalanced ICU data)",
                     fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"PR curves saved to {output_path}")
        plt.close()
