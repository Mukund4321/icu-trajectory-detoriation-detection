"""
Unit Tests
==========
Comprehensive test suite for all modules.

Run with:
    pytest tests.py -v
or simply:
    python tests.py
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import generate_synthetic_icu_data, ICUDataLoader
from src.preprocessing import ICUPreprocessor
from src.feature_engineering import TrajectoryFeatureEngineer
from src.ml_models import MLBaselines
from src.dl_models import LSTMTrajectoryModel, LSTMTrainer
from src.trajectory_logic import TrajectoryBasedDetector
from src.evaluation import TrajectoryEvaluator


def test_data_generation():
    """Test synthetic data generation."""
    print("\n[TEST] Data Generation...")
    
    df = generate_synthetic_icu_data(
        num_patients=10,
        records_per_patient=50,
        random_seed=42
    )
    
    # Assertions
    assert len(df) == 500, "Should have 500 records"
    assert df['patient_id'].nunique() == 10, "Should have 10 patients"
    assert set(df['label'].unique()).issubset({0, 1}), "Labels should be 0 or 1"
    assert df[['heart_rate', 'systolic_bp']].notna().all().all(), "No NaN in vitals"
    
    print("  ✓ Data generation passed")


def test_data_loader():
    """Test ICUDataLoader."""
    print("\n[TEST] Data Loader...")
    
    # Generate test data
    df = generate_synthetic_icu_data(num_patients=5, records_per_patient=30)
    test_path = Path('/tmp/test_icu.csv')
    df.to_csv(test_path, index=False)
    
    loader = ICUDataLoader()
    
    # Test loading
    df_loaded = loader.load_csv(str(test_path))
    assert len(df_loaded) == len(df), "Should load correct number of records"
    
    # Test validation
    df_validated = loader.validate_columns(df_loaded)
    assert all(col in df_validated.columns for col in loader.REQUIRED_COLUMNS)
    
    # Test formatting
    df_formatted = loader.format_timestamps(df_validated)
    assert df_formatted['timestamp'].dtype == 'datetime64[ns]'
    
    # Test sorting
    df_sorted = loader.sort_by_trajectory(df_formatted)
    for pid in df_sorted['patient_id'].unique():
        patient_times = df_sorted[df_sorted['patient_id'] == pid]['timestamp'].values
        assert all(patient_times[i] <= patient_times[i+1] for i in range(len(patient_times)-1))
    
    # Cleanup
    test_path.unlink()
    
    print("  ✓ Data loader passed")


def test_preprocessing():
    """Test ICUPreprocessor."""
    print("\n[TEST] Preprocessing...")
    
    df = generate_synthetic_icu_data(num_patients=5, records_per_patient=50)
    df = df.sort_values(['patient_id', 'timestamp'])
    
    preprocessor = ICUPreprocessor(resample_interval='5min', smoothing_window=3)
    df_processed, stats = preprocessor.process(df)
    
    # Assertions
    assert len(df_processed) > 0, "Should have output data"
    assert stats['num_patients'] == 5, "Should preserve patient count"
    assert 'normalization_stats' in stats, "Should track normalization"
    
    # Check normalization
    for pid, vital_stats in stats['normalization_stats'].items():
        for vital_name, v_stats in vital_stats.items():
            assert 'mean' in v_stats and 'std' in v_stats
    
    print("  ✓ Preprocessing passed")


def test_feature_engineering():
    """Test TrajectoryFeatureEngineer."""
    print("\n[TEST] Feature Engineering...")
    
    df = generate_synthetic_icu_data(num_patients=5, records_per_patient=50)
    df = df.sort_values(['patient_id', 'timestamp'])
    
    preprocessor = ICUPreprocessor()
    df_clean, _ = preprocessor.process(df)
    
    # Test sliding windows
    fe = TrajectoryFeatureEngineer(window_size=12, window_step=1)
    X_ml, y_ml, pids = fe.create_sliding_windows(df_clean)
    
    assert len(X_ml) > 0, "Should have feature windows"
    assert len(X_ml) == len(y_ml), "X and y should match"
    assert X_ml.shape[1] == len(fe.get_feature_names()), "Feature count mismatch"
    
    # Test LSTM sequences
    X_lstm, y_lstm = fe.create_sequence_data(df_clean, seq_length=12)
    
    assert X_lstm.shape[0] > 0, "Should have sequences"
    assert X_lstm.shape[-1] == 6, "Should have 6 vital features"
    assert len(X_lstm) == len(y_lstm), "Sequences and labels should match"
    
    print("  ✓ Feature engineering passed")


def test_ml_models():
    """Test MLBaselines."""
    print("\n[TEST] Machine Learning Models...")
    
    df = generate_synthetic_icu_data(num_patients=20, records_per_patient=40)
    df = df.sort_values(['patient_id', 'timestamp'])
    
    preprocessor = ICUPreprocessor()
    df_clean, _ = preprocessor.process(df)
    
    fe = TrajectoryFeatureEngineer()
    X, y, _ = fe.create_sliding_windows(df_clean)
    
    ml = MLBaselines()
    results = ml.train_and_evaluate(X, y)
    
    # Assertions
    assert 'LogisticRegression' in results, "Should have LR model"
    assert 'RandomForest' in results, "Should have RF model"
    
    for model_name, result in results.items():
        metrics = result['metrics']
        assert 'accuracy' in metrics and 0 <= metrics['accuracy'] <= 1
        assert 'precision' in metrics and 0 <= metrics['precision'] <= 1
        assert 'recall' in metrics and 0 <= metrics['recall'] <= 1
        assert 'f1' in metrics and 0 <= metrics['f1'] <= 1
        assert 'auc' in metrics and 0 <= metrics['auc'] <= 1
        assert 'confusion_matrix' in metrics
    
    # Test feature importance
    feature_names = fe.get_feature_names()
    importance = ml.get_feature_importance(feature_names)
    assert len(importance) == 2, "Should have importance for 2 models"
    
    print("  ✓ ML models passed")


def test_lstm_model():
    """Test LSTM model architecture and training."""
    print("\n[TEST] LSTM Model...")
    
    # Test model creation
    model = LSTMTrajectoryModel(input_size=6, hidden_size=32, num_layers=2)
    assert sum(p.numel() for p in model.parameters()) > 0, "Model should have parameters"
    
    # Test forward pass
    batch = torch.randn(16, 12, 6)  # (batch, seq_len, features)
    output = model(batch)
    assert output.shape == (16, 1), "Output should be (batch, 1)"
    
    # Test trainer
    df = generate_synthetic_icu_data(num_patients=20, records_per_patient=40)
    df = df.sort_values(['patient_id', 'timestamp'])
    
    preprocessor = ICUPreprocessor()
    df_clean, _ = preprocessor.process(df)
    
    fe = TrajectoryFeatureEngineer()
    X, y = fe.create_sequence_data(df_clean, seq_length=12)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    trainer = LSTMTrainer(model)
    training_info = trainer.train(X_train, y_train, X_val, y_val, epochs=5, patience=3)
    
    assert 'train_losses' in training_info
    assert 'val_losses' in training_info
    assert len(training_info['train_losses']) > 0
    
    # Test evaluation
    metrics = trainer.evaluate(X_test, y_test)
    assert 'accuracy' in metrics and 0 <= metrics['accuracy'] <= 1
    assert 'auc' in metrics and 0 <= metrics['auc'] <= 1
    
    print("  ✓ LSTM model passed")


def test_trajectory_detector():
    """Test TrajectoryBasedDetector."""
    print("\n[TEST] Trajectory Detector...")
    
    df = generate_synthetic_icu_data(num_patients=10, records_per_patient=50)
    df = df.sort_values(['patient_id', 'timestamp'])
    
    preprocessor = ICUPreprocessor()
    df_clean, _ = preprocessor.process(df)
    
    detector = TrajectoryBasedDetector()
    df_with_flags = detector.flag_deterioration(df_clean)
    
    # Assertions
    assert 'predicted_deterioration' in df_with_flags.columns
    assert 'instability_score' in df_with_flags.columns
    assert set(df_with_flags['predicted_deterioration'].unique()).issubset({0, 1})
    assert (df_with_flags['instability_score'] >= 0).all()
    
    # Test comparison
    metrics = detector.compare_predictions(df_with_flags)
    assert 'accuracy' in metrics
    assert 'confusion_matrix' in metrics
    
    print("  ✓ Trajectory detector passed")


def test_evaluator():
    """Test TrajectoryEvaluator."""
    print("\n[TEST] Evaluator...")
    
    df = generate_synthetic_icu_data(num_patients=10, records_per_patient=50)
    df = df.sort_values(['patient_id', 'timestamp'])
    
    preprocessor = ICUPreprocessor()
    df_clean, _ = preprocessor.process(df)
    
    fe = TrajectoryFeatureEngineer()
    X, y, _ = fe.create_sliding_windows(df_clean)
    
    ml = MLBaselines()
    ml_results = ml.train_and_evaluate(X, y)
    
    evaluator = TrajectoryEvaluator()
    
    # Test plotting functions (should not error)
    try:
        evaluator.plot_roc_curves(ml_results)
        evaluator.plot_confusion_matrices(ml_results)
        evaluator.plot_feature_importance(ml.get_feature_importance(fe.get_feature_names()))
        
        print("  ✓ Evaluator passed")
    except Exception as e:
        print(f"  ✗ Evaluator failed: {e}")
        raise


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RUNNING UNIT TESTS")
    print("=" * 70)
    
    tests = [
        test_data_generation,
        test_data_loader,
        test_preprocessing,
        test_feature_engineering,
        test_ml_models,
        test_lstm_model,
        test_trajectory_detector,
        test_evaluator
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test_func.__name__} FAILED: {str(e)}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
