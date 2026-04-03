"""
ML Models Module
================
Implements baseline machine learning models for trajectory classification.
Phase 2: Added XGBoost, LightGBM with automatic class-imbalance handling.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    average_precision_score, precision_recall_curve
)


class MLBaselines:
    """Train and evaluate baseline ML models."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize ML baselines.
        
        Parameters:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray,
                     patient_ids: np.ndarray = None,
                     test_size: float = 0.2
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/test sets and scale features.

        When patient_ids is provided, splits by patient (not by window).
        This is critical: splitting randomly by window means the model sees
        windows from the same patient in both train and test, inflating AUC
        by ~0.5 due to per-patient z-score pattern memorisation.

        Parameters:
            X: Feature matrix (windows)
            y: Labels
            patient_ids: Patient ID for each window. If provided, split is
                         done at patient level to prevent data leakage.
            test_size: Fraction for test set
        """
        print("\n" + "=" * 60)
        print("ML BASELINES")
        print("=" * 60)

        if patient_ids is not None:
            # Patient-level split — the correct approach
            unique_patients = np.unique(patient_ids)
            np.random.seed(self.random_seed)
            perm = np.random.permutation(len(unique_patients))
            n_test = max(1, int(len(unique_patients) * test_size))
            test_pids  = set(unique_patients[perm[:n_test]])
            train_pids = set(unique_patients[perm[n_test:]])

            train_mask = np.array([pid in train_pids for pid in patient_ids])
            test_mask  = ~train_mask

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            print(f"\nPatient-level split ({len(train_pids)} train / {len(test_pids)} test patients):")
        else:
            # Fallback: window-level split (used when no patient_ids available)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_seed, stratify=y
            )
            print(f"\nWindow-level split ({int(100*(1-test_size))} / {int(100*test_size)}):")

        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)

        print(f"  Train: {len(X_train)} windows  pos={np.sum(y_train):.0f} ({100*np.mean(y_train):.1f}%)")
        print(f"  Test:  {len(X_test)} windows   pos={np.sum(y_test):.0f}  ({100*np.mean(y_test):.1f}%)")

        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """
        Train logistic regression model.
        
        Parameters:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        print("\n[1] Logistic Regression")
        print("-" * 40)
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_seed,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        print(f"? Model trained")
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Train random forest with balanced class weights."""
        print("\n[2] Random Forest")
        print("-" * 40)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=self.random_seed,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        print(f"Trained with {model.n_estimators} trees")
        return model

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train XGBoost with scale_pos_weight to handle class imbalance.

        scale_pos_weight = neg_count / pos_count tells XGBoost to penalise
        missing a positive (sepsis) case proportionally to how rare it is.
        """
        from xgboost import XGBClassifier

        print("\n[3] XGBoost")
        print("-" * 40)

        pos = int(np.sum(y_train == 1))
        neg = int(np.sum(y_train == 0))
        scale_pos_weight = neg / pos if pos > 0 else 1.0
        print(f"  Class ratio neg/pos = {scale_pos_weight:.1f}  (auto scale_pos_weight)")

        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_seed,
            n_jobs=-1,
            eval_metric='aucpr',
            verbosity=0,
            use_label_encoder=False
        )
        model.fit(X_train, y_train)
        print(f"Trained with {model.n_estimators} trees")
        return model

    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train LightGBM with is_unbalance flag for class imbalance."""
        from lightgbm import LGBMClassifier

        print("\n[4] LightGBM")
        print("-" * 40)

        pos = int(np.sum(y_train == 1))
        neg = int(np.sum(y_train == 0))
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        model = LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_seed,
            n_jobs=-1,
            verbosity=-1
        )
        model.fit(X_train, y_train)
        print(f"Trained with {model.n_estimators} trees")
        return model
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                       model_name: str) -> Dict[str, Any]:
        """
        Evaluate model on test set. Reports AUC-ROC, AUPRC, F1, Recall.
        AUPRC (Area Under Precision-Recall Curve) is more informative than
        AUC-ROC for imbalanced datasets like ICU sepsis data (~2-3% positive).
        """
        y_pred       = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy':         accuracy_score(y_test, y_pred),
            'precision':        precision_score(y_test, y_pred, zero_division=0),
            'recall':           recall_score(y_test, y_pred, zero_division=0),
            'f1':               f1_score(y_test, y_pred, zero_division=0),
            'auc':              roc_auc_score(y_test, y_pred_proba),
            'auprc':            average_precision_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_curve':        roc_curve(y_test, y_pred_proba),
            'pr_curve':         precision_recall_curve(y_test, y_pred_proba),
            'y_test':           y_test,
            'y_pred':           y_pred,
            'y_pred_proba':     y_pred_proba,
        }

        print(f"\n{model_name} Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc']:.4f}")
        print(f"  AUPRC:     {metrics['auprc']:.4f}  <- key metric for imbalanced data")

        cm = metrics['confusion_matrix']
        print(f"  Confusion Matrix:  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

        return metrics
    
    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray,
                           patient_ids: np.ndarray = None,
                           test_size: float = 0.2) -> Dict[str, Dict[str, Any]]:
        """
        Full ML pipeline: prepare data, train all models, evaluate.
        Trains: LogisticRegression, RandomForest, XGBoost, LightGBM.
        Pass patient_ids for patient-level split (strongly recommended).
        """
        X_train, X_test, y_train, y_test = self.prepare_data(X, y, patient_ids, test_size)
        results = {}

        for name, train_fn in [
            ('LogisticRegression', self.train_logistic_regression),
            ('RandomForest',       self.train_random_forest),
            ('XGBoost',            self.train_xgboost),
            ('LightGBM',           self.train_lightgbm),
        ]:
            model = train_fn(X_train, y_train)
            results[name] = {
                'model':   model,
                'metrics': self.evaluate_model(model, X_test, y_test, name)
            }
            self.models[name] = model

        self.results = results
        return results
    
    def get_feature_importance(self, feature_names: list) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance from trained models.
        
        Parameters:
            feature_names: List of feature names
            
        Returns:
            Dictionary with importance DataFrames
        """
        importance_dict = {}
        
        # Logistic Regression coefficients
        if 'LogisticRegression' in self.models:
            lr = self.models['LogisticRegression']
            lr_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(lr.coef_[0])
            }).sort_values('importance', ascending=False)
            importance_dict['LogisticRegression'] = lr_importance
        
        # Tree-based feature importance (RF, XGBoost, LightGBM)
        for name in ('RandomForest', 'XGBoost', 'LightGBM'):
            if name in self.models:
                imp = pd.DataFrame({
                    'feature':    feature_names,
                    'importance': self.models[name].feature_importances_
                }).sort_values('importance', ascending=False)
                importance_dict[name] = imp

        return importance_dict
    
    def compare_models(self) -> pd.DataFrame:
        """
        Create comparison table of model performance.
        
        Returns:
            DataFrame with model comparison metrics
        """
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        
        comparison = []

        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison.append({
                'Model':     model_name,
                'Accuracy':  round(metrics['accuracy'],  4),
                'Precision': round(metrics['precision'], 4),
                'Recall':    round(metrics['recall'],    4),
                'F1-Score':  round(metrics['f1'],        4),
                'AUC-ROC':   round(metrics['auc'],       4),
                'AUPRC':     round(metrics.get('auprc', 0), 4),
            })
        
        df_comparison = pd.DataFrame(comparison)
        print("\n" + df_comparison.to_string(index=False))
        
        return df_comparison
