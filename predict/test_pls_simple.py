"""
Simple test for PLS predictor logic
Tests basic functionality without running the full model
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all imports work"""
    try:
        from pls_predictor import PLSPredictor, train_and_save_model, predict_formula_properties
        print("✅ PLS predictor imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_formula_extractor():
    """Test formula extraction logic"""
    try:
        from pls_predictor import FormulaExtractor
        import pandas as pd
        
        # Create mock data similar to datasets
        mock_data = pd.DataFrame({
            'is_heterostructure': [1, 0],
            'is_supported': [0, 1],
            'Ti': [0.6, 0.0],
            'C': [0.4, 0.5],
            'O': [0.0, 0.5],
            'target': [1, 0]
        })
        
        extractor = FormulaExtractor()
        formulas = extractor.extract_formula_from_features(mock_data)
        
        print(f"✅ Formula extraction test passed: {formulas}")
        return True
        
    except Exception as e:
        print(f"❌ Formula extraction test failed: {e}")
        return False

def test_basic_structure():
    """Test basic PLSPredictor structure"""
    try:
        from pls_predictor import PLSPredictor
        
        # Create predictor without training
        predictor = PLSPredictor(datasets_dir="./datasets", use_matminer=False)
        
        print(f"✅ PLSPredictor creation successful")
        print(f"   Datasets dir: {predictor.datasets_dir}")
        print(f"   Use matminer: {predictor.use_matminer}")
        print(f"   Is trained: {predictor.is_trained}")
        
        return True
        
    except Exception as e:
        print(f"❌ PLSPredictor creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing PLS Predictor Components")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Formula Extractor Test", test_formula_extractor),
        ("Basic Structure Test", test_basic_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"   Test failed")
        except Exception as e:
            print(f"   Test error: {e}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PLS predictor structure is correct.")
    else:
        print("⚠️ Some tests failed. Check dependencies and code structure.")

if __name__ == "__main__":
    main() 