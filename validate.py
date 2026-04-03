"""
Test & Validation Script
========================
Quick validation of project setup and dependency installation.
"""

import sys
import os


def check_dependencies():
    """Check if all required packages are installed."""
    print("=" * 70)
    print("DEPENDENCY CHECK")
    print("=" * 70)
    
    dependencies = {
        'pandas': 'Data processing',
        'numpy': 'Numerical computing',
        'scipy': 'Scientific computing',
        'sklearn': 'Machine learning',
        'torch': 'Deep learning (PyTorch)',
        'matplotlib': 'Plotting',
        'seaborn': 'Statistical visualization'
    }
    
    missing = []
    installed = []
    
    for package, description in dependencies.items():
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            installed.append((package, description))
            print(f"✓ {package:15} - {description}")
        except ImportError:
            missing.append((package, description))
            print(f"✗ {package:15} - {description}")
    
    print("-" * 70)
    
    if missing:
        print(f"\n❌ Missing {len(missing)} dependencies:")
        for pkg, desc in missing:
            print(f"   pip install {pkg}")
        return False
    else:
        print(f"\n✅ All dependencies installed!")
        return True


def check_project_structure():
    """Verify project folder structure."""
    print("\n" + "=" * 70)
    print("PROJECT STRUCTURE CHECK")
    print("=" * 70)
    
    required_dirs = [
        'data',
        'data/raw',
        'data/processed',
        'src',
        'models',
        'results'
    ]
    
    required_files = [
        'main.py',
        'config.py',
        'requirements.txt',
        'README.md',
        'src/__init__.py',
        'src/data_loader.py',
        'src/preprocessing.py',
        'src/feature_engineering.py',
        'src/ml_models.py',
        'src/dl_models.py',
        'src/trajectory_logic.py',
        'src/evaluation.py'
    ]
    
    all_ok = True
    
    print("\nDirectories:")
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (MISSING)")
            all_ok = False
    
    print("\nFiles:")
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (MISSING)")
            all_ok = False
    
    print("-" * 70)
    
    if all_ok:
        print("✅ Project structure is correct!")
    else:
        print("❌ Some files/directories are missing")
    
    return all_ok


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "=" * 70)
    print("MODULE IMPORT TEST")
    print("=" * 70)
    
    modules = [
        'src.data_loader',
        'src.preprocessing',
        'src.feature_engineering',
        'src.ml_models',
        'src.dl_models',
        'src.trajectory_logic',
        'src.evaluation'
    ]
    
    all_ok = True
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except Exception as e:
            print(f"✗ {module}: {str(e)}")
            all_ok = False
    
    print("-" * 70)
    
    if all_ok:
        print("✅ All modules imported successfully!")
    else:
        print("❌ Some modules failed to import")
    
    return all_ok


def test_pytorch():
    """Test PyTorch and CUDA availability."""
    print("\n" + "=" * 70)
    print("PYTORCH & DEVICE CHECK")
    print("=" * 70)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
        else:
            print(f"⚠ CUDA not available (will use CPU)")
        
        print("-" * 70)
        print("✅ PyTorch is ready!")
        return True
    except Exception as e:
        print(f"❌ PyTorch error: {str(e)}")
        return False


def main():
    """Run all validation checks."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  ICU TRAJECTORY DETERIORATION DETECTION - VALIDATION SCRIPT".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    results = []
    
    # Run checks
    results.append(("Dependencies", check_dependencies()))
    results.append(("Project Structure", check_project_structure()))
    results.append(("Module Imports", test_imports()))
    results.append(("PyTorch", test_pytorch()))
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:30} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 70)
    
    if all_passed:
        print("\n✅ All checks passed! Ready to run: python main.py")
        return 0
    else:
        print("\n❌ Some checks failed. Fix issues above before running main.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
