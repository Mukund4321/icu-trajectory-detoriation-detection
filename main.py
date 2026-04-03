"""
Main Pipeline
=============
Orchestrates the complete ICU trajectory deterioration detection system.

Workflow:
1. Generate/Load synthetic ICU data
2. Preprocess data (resampling, normalization, smoothing)
3. Feature engineering (sliding windows, trends)
4. Train ML baselines (Logistic Regression, Random Forest)
5. Train LSTM deep learning model
6. Run trajectory-based detection logic
7. Evaluate all approaches and generate visualizations
"""

import sys
import numpy as np
import pandas as pd
import torch
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Import project modules
from src.data_loader import ICUDataLoader, generate_synthetic_icu_data
from src.data_adapter import load_physionet_2019, patient_level_split, apply_deterioration_labels
from src.preprocessing import ICUPreprocessor
from src.feature_engineering import TrajectoryFeatureEngineer
from src.ml_models import MLBaselines
from src.dl_models import (
    LSTMTrajectoryModel, LSTMTrainer,
    GRUTrajectoryModel, GRUTrainer,
    BiLSTMTrajectoryModel, BiLSTMTrainer,
)
from src.trajectory_logic import TrajectoryBasedDetector
from src.evaluation import TrajectoryEvaluator


def main():
    """
    Execute complete pipeline for ICU trajectory deterioration detection.
    """
    
    print("\n" + "=" * 70)
    print("ICU TRAJECTORY DETERIORATION DETECTION SYSTEM")
    print("Health Trajectory Modeling for Early Physiological Deterioration")
    print("=" * 70)
    
    # Setup random seeds for reproducibility
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Setup directories
    data_dir = Path('data')
    raw_data_dir = data_dir / 'raw'
    processed_data_dir = data_dir / 'processed'
    results_dir = Path('results')
    
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== 1. DATA LOADING ==========
    print("\n" + ">" * 35)
    print("STEP 1: DATA LOADING")
    print(">" * 35)

    # Try PhysioNet 2019 first (real longitudinal ICU data).
    # Falls back to synthetic data if not downloaded yet.
    try:
        df_raw = load_physionet_2019(str(raw_data_dir))
        data_source = 'PhysioNet 2019'
    except FileNotFoundError as e:
        print(f"\n[INFO] {e}")
        print("\nFalling back to synthetic ICU data...")
        raw_csv_path = raw_data_dir / 'synthetic_icu_data.csv'
        if not raw_csv_path.exists():
            generate_synthetic_icu_data(
                num_patients=50,
                records_per_patient=100,
                output_path=str(raw_csv_path),
                random_seed=RANDOM_SEED
            )
        loader = ICUDataLoader(random_seed=RANDOM_SEED)
        df_raw, _ = loader.process_raw_data(str(raw_csv_path))
        data_source = 'Synthetic'

    print(f"\nOK Data source: {data_source}")
    print(f"  Patients: {df_raw['patient_id'].nunique()}")
    print(f"  Records:  {len(df_raw)}")

    # Use vital-sign based deterioration labels instead of SepsisLabel.
    # Flags any row where HR, BP, SpO2, RR or Temp breaches clinical thresholds.
    # Produces ~15-25% positive rate vs 5% for sepsis — far more training signal.
    if data_source == 'PhysioNet 2019':
        df_raw = apply_deterioration_labels(df_raw)

    # ========== 1B. PATIENT-LEVEL SPLIT (before any preprocessing) ==========
    # Split patients into train/val/test BEFORE preprocessing so normalization
    # stats are computed only on training patients - no look-ahead leakage.
    print("\nSplitting by patient (not by window)...")
    df_train_raw, df_val_raw, df_test_raw = patient_level_split(
        df_raw, test_size=0.2, val_size=0.1, random_seed=RANDOM_SEED
    )

    # ========== 2. PREPROCESSING ==========
    print("\n" + ">" * 35)
    print("STEP 2: PREPROCESSING")
    print(">" * 35)

    # PhysioNet is natively hourly — resampling to 5min just inflates rows 12x
    # with identical ffill values, making window creation ~12x slower for zero gain.
    # Synthetic data was generated at 5min intervals so keep that.
    resample_interval = '1h' if data_source == 'PhysioNet 2019' else '5min'
    # At 1h intervals, window_size=12 = 12-hour window (appropriate for sepsis).
    # At 5min intervals, window_size=12 = 1-hour window (original design).
    window_size = 12

    preprocessor = ICUPreprocessor(resample_interval=resample_interval, smoothing_window=3)

    df_train_proc, _ = preprocessor.process(df_train_raw, baseline_window=12)
    df_val_proc, _  = preprocessor.process(df_val_raw,  baseline_window=12)
    df_test_proc, _ = preprocessor.process(df_test_raw, baseline_window=12)

    # Combine processed splits (with split tag for tracking)
    df_train_proc['_split'] = 'train'
    df_val_proc['_split']   = 'val'
    df_test_proc['_split']  = 'test'
    df_processed = pd.concat([df_train_proc, df_val_proc, df_test_proc], ignore_index=True)

    # Save processed data
    processed_csv_path = processed_data_dir / 'icu_data_processed.csv'
    df_processed.to_csv(processed_csv_path, index=False)
    print(f"OK Processed data saved to {processed_csv_path}")

    # ========== 3. FEATURE ENGINEERING ==========
    print("\n" + ">" * 35)
    print("STEP 3: FEATURE ENGINEERING")
    print(">" * 35)

    feature_engineer = TrajectoryFeatureEngineer(window_size=window_size, window_step=1)

    # Build features PER SPLIT to prevent window leakage across patient splits
    X_train_ml, y_train_ml, pids_train_ml = feature_engineer.create_sliding_windows(df_train_proc)
    X_val_ml,   y_val_ml,   pids_val_ml   = feature_engineer.create_sliding_windows(df_val_proc)
    X_test_ml,  y_test_ml,  _             = feature_engineer.create_sliding_windows(df_test_proc)

    # Concatenate train+val for ML models — patient_ids passed so internal
    # split is patient-level (not window-level) preventing memorisation leakage
    X_ml       = np.concatenate([X_train_ml, X_val_ml])
    y_ml       = np.concatenate([y_train_ml, y_val_ml])
    pids_ml    = np.concatenate([pids_train_ml, pids_val_ml])

    feature_names = feature_engineer.get_feature_names()
    print(f"Feature vector dimension: {len(feature_names)}")

    # LSTM sequences - also per split
    X_train_lstm, y_train_lstm = feature_engineer.create_sequence_data(df_train_proc, seq_length=12)
    X_val_lstm,   y_val_lstm   = feature_engineer.create_sequence_data(df_val_proc,   seq_length=12)
    X_test_lstm,  y_test_lstm  = feature_engineer.create_sequence_data(df_test_proc,  seq_length=12)

    # ========== 4. MACHINE LEARNING BASELINES ==========
    print("\n" + ">" * 35)
    print("STEP 4: MACHINE LEARNING BASELINES")
    print(">" * 35)

    # MLBaselines.train_and_evaluate does its own 80/20 split internally.
    # We pass train+val windows; test_ml is held out for honest final eval.
    ml_trainer = MLBaselines(random_seed=RANDOM_SEED)
    ml_results = ml_trainer.train_and_evaluate(X_ml, y_ml,
                                               patient_ids=pids_ml, test_size=0.2)

    # Final eval on held-out test patients
    X_test_ml_scaled = ml_trainer.scaler.transform(X_test_ml)
    for name, result in ml_results.items():
        test_metrics = ml_trainer.evaluate_model(
            result['model'], X_test_ml_scaled, y_test_ml, f"{name} [held-out test]"
        )
        result['test_metrics'] = test_metrics

    feature_importance = ml_trainer.get_feature_importance(feature_names)
    ml_comparison = ml_trainer.compare_models()

    # Save ML models + scaler to disk so backend.py can load them for live predictions
    import joblib
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    for model_name, result in ml_results.items():
        joblib.dump(result['model'], models_dir / f'{model_name.lower()}_model.pkl')
    joblib.dump(ml_trainer.scaler, models_dir / 'ml_scaler.pkl')
    joblib.dump(feature_names, models_dir / 'feature_names.pkl')
    print(f"OK ML models saved to {models_dir}/")

    # ========== 5. DEEP LEARNING - LSTM ==========
    print("\n" + ">" * 35)
    print("STEP 5: DEEP LEARNING - LSTM MODEL")
    print(">" * 35)

    # Use patient-level splits directly - no further random splitting
    X_train, y_train = X_train_lstm, y_train_lstm
    X_val,   y_val   = X_val_lstm,   y_val_lstm
    X_test,  y_test  = X_test_lstm,  y_test_lstm

    print(f"\nData shapes (patient-level split):")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    
    # Compute pos_weight to fix class imbalance (e.g. 97:3 ratio -> pos_weight=32)
    # Without this, LSTM learns "predict all-negative" which gives high accuracy
    # but zero recall — useless for detecting sepsis.
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    lstm_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"\nClass imbalance: {n_neg} negative, {n_pos} positive -> pos_weight={lstm_pos_weight:.1f}")

    input_size = X_train.shape[2]
    lstm_model = LSTMTrajectoryModel(input_size=input_size, hidden_size=128,
                                     num_layers=2, dropout=0.3)

    trainer = LSTMTrainer(lstm_model, learning_rate=0.0005, random_seed=RANDOM_SEED,
                          pos_weight=lstm_pos_weight)
    training_info = trainer.train(X_train, y_train, X_val, y_val,
                                  epochs=100, batch_size=256, patience=15)
    
    # Evaluate LSTM
    lstm_metrics = trainer.evaluate(X_test, y_test)
    
    # Save LSTM model
    model_path = Path('models') / 'lstm_model.pt'
    model_path.parent.mkdir(exist_ok=True)
    trainer.save_model(str(model_path))
    
    # ========== 5B. DEEP LEARNING - GRU MODEL ==========
    print("\n" + ">" * 35)
    print("STEP 5B: DEEP LEARNING - GRU MODEL")
    print(">" * 35)
    
    # Create and train GRU
    gru_model = GRUTrajectoryModel(input_size=input_size, hidden_size=128,
                                   num_layers=2, dropout=0.3)

    gru_trainer = GRUTrainer(gru_model, learning_rate=0.0005, random_seed=RANDOM_SEED,
                             pos_weight=lstm_pos_weight)
    gru_training_info = gru_trainer.train(X_train, y_train, X_val, y_val,
                                          epochs=100, batch_size=256, patience=15)
    
    # Evaluate GRU
    gru_metrics = gru_trainer.evaluate(X_test, y_test)
    
    # Save GRU model
    gru_model_path = Path('models') / 'gru_model.pt'
    gru_model_path.parent.mkdir(exist_ok=True)
    gru_trainer.save_model(str(gru_model_path))

    # ========== 5C. DEEP LEARNING - BILSTM MODEL ==========
    print("\n" + ">" * 35)
    print("STEP 5C: DEEP LEARNING - BILSTM MODEL")
    print(">" * 35)
    
    bilstm_model = BiLSTMTrajectoryModel(input_size=input_size, hidden_size=128,
                                         num_layers=2, dropout=0.3)

    bilstm_trainer = BiLSTMTrainer(bilstm_model, learning_rate=0.0005, random_seed=RANDOM_SEED,
                                   pos_weight=lstm_pos_weight)
    bilstm_training_info = bilstm_trainer.train(X_train, y_train, X_val, y_val,
                                                epochs=100, batch_size=256, patience=15)
    
    bilstm_metrics = bilstm_trainer.evaluate(X_test, y_test)
    
    bilstm_model_path = Path('models') / 'bilstm_model.pt'
    bilstm_model_path.parent.mkdir(exist_ok=True)
    bilstm_trainer.save_model(str(bilstm_model_path))

    # ========== 6. TRAJECTORY-BASED DETECTION ==========
    print("\n" + ">" * 35)
    print("STEP 6: TRAJECTORY-BASED DETERIORATION DETECTION")
    print(">" * 35)
    
    trajectory_detector = TrajectoryBasedDetector(
        baseline_window=12,
        instability_window=3,
        persistence_threshold=2
    )
    
    df_with_predictions = trajectory_detector.flag_deterioration(df_processed)
    trajectory_metrics = trajectory_detector.compare_predictions(df_with_predictions)
    
    # Try to identify phenotypes
    phenotypes = trajectory_detector.identify_deterioration_phenotypes(
        df_with_predictions, num_clusters=3
    )
    
    # ========== 7. EVALUATION & VISUALIZATION ==========
    print("\n" + ">" * 35)
    print("STEP 7: EVALUATION & VISUALIZATION")
    print(">" * 35)
    
    evaluator = TrajectoryEvaluator(results_dir=str(results_dir))
    
    # ========== 6B. WEIGHTED ENSEMBLE ==========
    print("\n" + ">" * 35)
    print("STEP 6B: WEIGHTED ENSEMBLE")
    print(">" * 35)

    # Ensemble: RF (40%) + LR (35%) + LSTM (25%) on held-out test patients.
    # RF and LR are consistently the best-performing held-out models.
    # LSTM adds temporal pattern signal that tabular models miss.
    try:
        rf_proba   = ml_results['RandomForest']['test_metrics']['y_pred_proba']
        lr_proba   = ml_results['LogisticRegression']['test_metrics']['y_pred_proba']
        lstm_proba = lstm_metrics.get('y_pred_proba', None)

        if lstm_proba is not None and len(rf_proba) == len(lstm_proba):
            ensemble_proba = 0.40 * rf_proba + 0.35 * lr_proba + 0.25 * lstm_proba
            ensemble_pred  = (ensemble_proba > 0.5).astype(int)
            y_test_ens     = ml_results['RandomForest']['test_metrics']['y_test']

            from sklearn.metrics import (accuracy_score, precision_score,
                                         recall_score, f1_score,
                                         roc_auc_score, roc_curve,
                                         average_precision_score,
                                         precision_recall_curve, confusion_matrix)

            ensemble_metrics = {
                'accuracy':         accuracy_score(y_test_ens, ensemble_pred),
                'precision':        precision_score(y_test_ens, ensemble_pred, zero_division=0),
                'recall':           recall_score(y_test_ens, ensemble_pred, zero_division=0),
                'f1':               f1_score(y_test_ens, ensemble_pred, zero_division=0),
                'auc':              roc_auc_score(y_test_ens, ensemble_proba),
                'auprc':            average_precision_score(y_test_ens, ensemble_proba),
                'confusion_matrix': confusion_matrix(y_test_ens, ensemble_pred),
                'roc_curve':        roc_curve(y_test_ens, ensemble_proba),
                'pr_curve':         precision_recall_curve(y_test_ens, ensemble_proba),
                'y_test':           y_test_ens,
                'y_pred':           ensemble_pred,
                'y_pred_proba':     ensemble_proba,
            }
            print(f"\nEnsemble (RF 40% + LR 35% + LSTM 25%) on held-out test patients:")
            print(f"  AUC-ROC: {ensemble_metrics['auc']:.4f}")
            print(f"  AUPRC:   {ensemble_metrics['auprc']:.4f}")
            print(f"  F1:      {ensemble_metrics['f1']:.4f}")
            print(f"  Recall:  {ensemble_metrics['recall']:.4f}")
            cm = ensemble_metrics['confusion_matrix']
            print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
        else:
            print("  Skipping ensemble: test set size mismatch between XGBoost and LSTM.")
            ensemble_metrics = None
    except Exception as e:
        print(f"  Ensemble skipped: {e}")
        ensemble_metrics = None

    # Combine all model results
    all_results = {
        **ml_results,
        'LSTM': {'model': lstm_model, 'metrics': lstm_metrics},
        'GRU':  {'model': gru_model,  'metrics': gru_metrics},
    }
    if ensemble_metrics is not None:
        all_results['Ensemble'] = {'model': None, 'metrics': ensemble_metrics}

    # Generate plots
    evaluator.plot_roc_curves(
        all_results,
        output_path=str(results_dir / 'roc_curves.png')
    )

    evaluator.plot_confusion_matrices(
        all_results,
        output_path=str(results_dir / 'confusion_matrices.png')
    )

    evaluator.plot_training_history(
        training_info['train_losses'],
        training_info['val_losses'],
        output_path=str(results_dir / 'lstm_training_history.png')
    )

    evaluator.plot_feature_importance(
        feature_importance,
        top_n=10,
        output_path=str(results_dir / 'feature_importance.png')
    )

    evaluator.plot_trajectory_examples(
        df_with_predictions,
        predictions={'LogisticRegression': ml_results['LogisticRegression']['metrics']['y_pred']},
        sample_patients=3,
        output_path=str(results_dir / 'trajectory_examples.png')
    )

    # Precision-Recall curves (more meaningful than ROC for 2-3% positive rate)
    evaluator.plot_precision_recall_curves(
        all_results,
        output_path=str(results_dir / 'pr_curves.png')
    )

    # SHAP explainability — use best tree model (XGBoost preferred for SHAP speed)
    shap_model_name = 'XGBoost' if 'XGBoost' in ml_trainer.models else 'RandomForest'
    evaluator.plot_shap_summary(
        ml_trainer.models[shap_model_name],
        X_test_ml_scaled,
        feature_names,
        model_name=shap_model_name,
        output_path=str(results_dir / 'shap_summary.png')
    )

    # Generate summary report
    report = evaluator.generate_summary_report(
        all_results,
        trajectory_metrics,
        output_path=str(results_dir / 'summary_report.txt')
    )

    print("\n" + report)
    
    # ========== 8. FINAL SUMMARY ==========
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE - SUMMARY")
    print("=" * 70)
    
    print("\n? RESULTS SUMMARY:")
    print("-" * 70)
    
    print("\nMachine Learning Models:")
    print(ml_comparison.to_string(index=False))
    
    print("\n\nLSTM Model Performance:")
    print(f"  Accuracy:  {lstm_metrics['accuracy']:.4f}")
    print(f"  Precision: {lstm_metrics['precision']:.4f}")
    print(f"  Recall:    {lstm_metrics['recall']:.4f}")
    print(f"  F1-Score:  {lstm_metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {lstm_metrics['auc']:.4f}")
    
    print("\n\nTrajectory-Based Detection:")
    print(f"  Accuracy:  {trajectory_metrics['accuracy']:.4f}")
    print(f"  Precision: {trajectory_metrics['precision']:.4f}")
    print(f"  Recall:    {trajectory_metrics['recall']:.4f}")
    print(f"  F1-Score:  {trajectory_metrics['f1']:.4f}")
    
    print("\n" + "=" * 70)
    print("? Output Files:")
    print("-" * 70)
    print(f"  Raw Data:        {raw_data_dir}")
    print(f"  Processed Data:  {processed_csv_path}")
    print(f"  Plots:           {results_dir}/")
    print(f"  LSTM Model:      {model_path}")
    print("=" * 70)
    
    print("\nOK Pipeline execution successful!")
    print("Review plots and results in the 'results/' directory.")


if __name__ == "__main__":
    main()
