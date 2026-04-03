"""
Quick Start Guide
=================
Fast way to get started with the ICU Trajectory system.
"""

import sys
from pathlib import Path

# Create required directories if they don't exist
Path('data/raw').mkdir(parents=True, exist_ok=True)
Path('data/processed').mkdir(parents=True, exist_ok=True)
Path('models').mkdir(parents=True, exist_ok=True)
Path('results').mkdir(parents=True, exist_ok=True)

def quick_start_ml_only():
    """
    Quick start: ML models only (fastest run time ~10 seconds).
    """
    print("\n" + "=" * 70)
    print("QUICK START: MACHINE LEARNING BASELINES")
    print("=" * 70)
    
    from src.data_loader import generate_synthetic_icu_data
    from src.preprocessing import ICUPreprocessor
    from src.feature_engineering import TrajectoryFeatureEngineer
    from src.ml_models import MLBaselines
    import numpy as np
    
    np.random.seed(42)
    
    # 1. Data
    print("\n[1/3] Generating data...")
    df = generate_synthetic_icu_data(num_patients=30, records_per_patient=60)
    
    # 2. Preprocessing
    print("\n[2/3] Preprocessing...")
    preprocessor = ICUPreprocessor()
    df_clean, _ = preprocessor.process(df)
    
    # 3. Features & ML
    print("\n[3/3] Training ML models...")
    fe = TrajectoryFeatureEngineer()
    X, y, _ = fe.create_sliding_windows(df_clean)
    
    ml = MLBaselines()
    results = ml.train_and_evaluate(X, y)
    
    print("\n? Quick start complete!")
    return results


def quick_start_lstm():
    """
    Quick start: LSTM only (medium run time ~30 seconds).
    """
    print("\n" + "=" * 70)
    print("QUICK START: LSTM DEEP LEARNING")
    print("=" * 70)
    
    from src.data_loader import generate_synthetic_icu_data
    from src.preprocessing import ICUPreprocessor
    from src.feature_engineering import TrajectoryFeatureEngineer
    from src.dl_models import LSTMTrajectoryModel, LSTMTrainer
    from sklearn.model_selection import train_test_split
    import torch
    import numpy as np
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. Data
    print("\n[1/4] Generating data...")
    df = generate_synthetic_icu_data(num_patients=30, records_per_patient=60)
    
    # 2. Preprocessing
    print("\n[2/4] Preprocessing...")
    preprocessor = ICUPreprocessor()
    df_clean, _ = preprocessor.process(df)
    
    # 3. Features
    print("\n[3/4] Feature engineering...")
    fe = TrajectoryFeatureEngineer()
    X, y = fe.create_sequence_data(df_clean, seq_length=10)
    
    # 4. LSTM
    print("\n[4/4] Training LSTM...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    model = LSTMTrajectoryModel(input_size=6, hidden_size=32, num_layers=1)
    trainer = LSTMTrainer(model)
    trainer.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=16, patience=5)
    metrics = trainer.evaluate(X_test, y_test)
    
    print("\n? LSTM training complete!")
    return metrics


def quick_start_full():
    """
    Full pipeline (longest run time ~2-3 minutes).
    Runs the complete system with all components.
    """
    print("\n" + "=" * 70)
    print("QUICK START: FULL PIPELINE")
    print("=" * 70)
    
    # Just import and run main
    from main import main
    main()


def interactive_menu():
    """Interactive menu for quick start options."""
    while True:
        print("\n" + "=" * 70)
        print("ICU TRAJECTORY - QUICK START OPTIONS")
        print("=" * 70)
        print("\n[1] ML Baselines Only (~10 seconds)")
        print("    - Logistic Regression + Random Forest")
        print("    - Fast, good for testing setup")
        print("\n[2] LSTM Deep Learning Only (~30 seconds)")
        print("    - LSTM model training and evaluation")
        print("    - Good for checking GPU/PyTorch setup")
        print("\n[3] Full Pipeline (~2-3 minutes)")
        print("    - Complete system: ML + LSTM + Trajectory Detection")
        print("    - Generates all plots and reports")
        print("\n[4] Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            quick_start_ml_only()
        elif choice == '2':
            quick_start_lstm()
        elif choice == '3':
            quick_start_full()
        elif choice == '4':
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    # If passing argument, run that specific option
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == 'ml':
            quick_start_ml_only()
        elif arg == 'lstm':
            quick_start_lstm()
        elif arg == 'full':
            quick_start_full()
        else:
            print(f"Unknown option: {arg}")
            print("Usage: python quick_start.py [ml|lstm|full]")
    else:
        # Interactive mode
        interactive_menu()
