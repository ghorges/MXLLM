"""
Test script for PLS Predictor
Comprehensive evaluation of prediction performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from pls_predictor import PLSPredictor, create_predictor

def generate_realistic_test_data(n_samples=200):
    """Generate realistic test data for microwave absorption materials"""
    print("🧪 Generating realistic test data...")
    
    # Common microwave absorption materials with realistic properties
    material_templates = [
        # Ferrites
        {"formula": "Fe3O4", "rl_base": -15, "eab_base": 6.5, "category": "ferrite"},
        {"formula": "NiFe2O4", "rl_base": -12, "eab_base": 5.8, "category": "ferrite"},
        {"formula": "CoFe2O4", "rl_base": -14, "eab_base": 6.2, "category": "ferrite"},
        {"formula": "ZnFe2O4", "rl_base": -10, "eab_base": 4.5, "category": "ferrite"},
        {"formula": "MnFe2O4", "rl_base": -11, "eab_base": 5.1, "category": "ferrite"},
        {"formula": "CuFe2O4", "rl_base": -9, "eab_base": 4.2, "category": "ferrite"},
        {"formula": "BaFe12O19", "rl_base": -18, "eab_base": 7.8, "category": "hexaferrite"},
        {"formula": "SrFe12O19", "rl_base": -16, "eab_base": 7.1, "category": "hexaferrite"},
        
        # Metal oxides
        {"formula": "TiO2", "rl_base": -8, "eab_base": 3.2, "category": "oxide"},
        {"formula": "ZnO", "rl_base": -3, "eab_base": 1.8, "category": "oxide"},
        {"formula": "Al2O3", "rl_base": -2, "eab_base": 1.2, "category": "oxide"},
        {"formula": "SiO2", "rl_base": -1.5, "eab_base": 0.9, "category": "oxide"},
        {"formula": "Fe2O3", "rl_base": -7, "eab_base": 2.8, "category": "oxide"},
        {"formula": "CuO", "rl_base": -6, "eab_base": 2.5, "category": "oxide"},
        {"formula": "NiO", "rl_base": -5, "eab_base": 2.1, "category": "oxide"},
        
        # Carbon materials
        {"formula": "C", "rl_base": -9, "eab_base": 4.1, "category": "carbon"},
        
        # Metals
        {"formula": "Fe", "rl_base": -11, "eab_base": 5.2, "category": "metal"},
        {"formula": "Ni", "rl_base": -13, "eab_base": 6.1, "category": "metal"},
        {"formula": "Co", "rl_base": -12, "eab_base": 5.7, "category": "metal"},
        
        # Composites (simplified formulas)
        {"formula": "FeNi", "rl_base": -10, "eab_base": 4.8, "category": "alloy"},
        {"formula": "FeCo", "rl_base": -11, "eab_base": 5.3, "category": "alloy"},
    ]
    
    formulas = []
    rl_values = []
    eab_values = []
    categories = []
    
    for i in range(n_samples):
        # Select random template
        template = np.random.choice(material_templates)
        
        # Add noise to base values
        rl_noise = np.random.normal(0, 2)  # ±2 dB variation
        eab_noise = np.random.normal(0, 0.8)  # ±0.8 GHz variation
        
        rl = template["rl_base"] + rl_noise
        eab = max(0.1, template["eab_base"] + eab_noise)  # EAB can't be negative
        
        formulas.append(template["formula"])
        rl_values.append(rl)
        eab_values.append(eab)
        categories.append(template["category"])
    
    print(f"✅ Generated {n_samples} samples")
    print(f"   RL range: {min(rl_values):.1f} to {max(rl_values):.1f} dB")
    print(f"   EAB range: {min(eab_values):.1f} to {max(eab_values):.1f} GHz")
    
    return formulas, rl_values, eab_values, categories


def evaluate_predictor(predictor, X_test, y_rl_test, y_eab_test, rl_test_encoded, eab_test_encoded):
    """Comprehensive evaluation of predictor performance"""
    print("\n📊 Evaluating predictor performance...")
    
    # Make predictions
    X_test_scaled = predictor.scaler.transform(X_test)
    rl_pred_encoded = predictor.rl_model.predict(X_test_scaled)
    eab_pred_encoded = predictor.eab_model.predict(X_test_scaled)
    
    # Calculate metrics
    rl_accuracy = accuracy_score(rl_test_encoded, rl_pred_encoded)
    rl_f1 = f1_score(rl_test_encoded, rl_pred_encoded, average='weighted')
    eab_accuracy = accuracy_score(eab_test_encoded, eab_pred_encoded)
    eab_f1 = f1_score(eab_test_encoded, eab_pred_encoded, average='weighted')
    
    print(f"🎯 RL Model Performance:")
    print(f"   Accuracy: {rl_accuracy:.3f}")
    print(f"   F1 Score: {rl_f1:.3f}")
    
    print(f"🎯 EAB Model Performance:")
    print(f"   Accuracy: {eab_accuracy:.3f}")
    print(f"   F1 Score: {eab_f1:.3f}")
    
    # Detailed classification reports
    print(f"\n📋 RL Classification Report:")
    rl_classes = predictor.rl_label_encoder.classes_
    print(classification_report(rl_test_encoded, rl_pred_encoded, 
                              target_names=rl_classes, zero_division=0))
    
    print(f"\n📋 EAB Classification Report:")
    eab_classes = predictor.eab_label_encoder.classes_
    print(classification_report(eab_test_encoded, eab_pred_encoded, 
                              target_names=eab_classes, zero_division=0))
    
    return {
        'rl_accuracy': rl_accuracy,
        'rl_f1': rl_f1,
        'eab_accuracy': eab_accuracy,
        'eab_f1': eab_f1,
        'rl_pred': rl_pred_encoded,
        'eab_pred': eab_pred_encoded
    }


def plot_confusion_matrices(predictor, rl_test_encoded, eab_test_encoded, rl_pred, eab_pred):
    """Plot confusion matrices for RL and EAB predictions"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # RL confusion matrix
        rl_cm = confusion_matrix(rl_test_encoded, rl_pred)
        rl_classes = predictor.rl_label_encoder.classes_
        sns.heatmap(rl_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=rl_classes, yticklabels=rl_classes, ax=ax1)
        ax1.set_title('RL Prediction Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # EAB confusion matrix
        eab_cm = confusion_matrix(eab_test_encoded, eab_pred)
        eab_classes = predictor.eab_label_encoder.classes_
        sns.heatmap(eab_cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=eab_classes, yticklabels=eab_classes, ax=ax2)
        ax2.set_title('EAB Prediction Confusion Matrix')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Confusion matrices saved as 'confusion_matrices.png'")
        
    except ImportError:
        print("⚠️ Matplotlib/Seaborn not available, skipping plots")


def test_cross_validation(formulas, rl_values, eab_values, cv_folds=5):
    """Perform cross-validation testing"""
    print(f"\n🔄 Performing {cv_folds}-fold cross-validation...")
    
    # Create predictor
    predictor = create_predictor(use_matminer=True, n_components=8)
    
    # Extract features
    X = predictor.feature_extractor.extract_features(formulas)
    
    # Prepare targets
    rl_array = np.array(rl_values)
    eab_array = np.array(eab_values)
    rl_classes, eab_classes = predictor._discretize_targets(rl_array, eab_array)
    
    # Encode labels
    rl_encoded = predictor.rl_label_encoder.fit_transform(rl_classes)
    eab_encoded = predictor.eab_label_encoder.fit_transform(eab_classes)
    
    # Scale features
    X_scaled = predictor.scaler.fit_transform(X)
    
    # Create pipelines for CV
    rl_pipeline = predictor._create_pls_pipeline(min(8, X_scaled.shape[1]))
    eab_pipeline = predictor._create_pls_pipeline(min(8, X_scaled.shape[1]))
    
    # Cross-validation
    rl_scores = cross_val_score(rl_pipeline, X_scaled, rl_encoded, cv=cv_folds, scoring='accuracy')
    eab_scores = cross_val_score(eab_pipeline, X_scaled, eab_encoded, cv=cv_folds, scoring='accuracy')
    
    print(f"🎯 Cross-validation Results:")
    print(f"   RL CV Accuracy: {rl_scores.mean():.3f} ± {rl_scores.std():.3f}")
    print(f"   EAB CV Accuracy: {eab_scores.mean():.3f} ± {eab_scores.std():.3f}")
    
    return rl_scores, eab_scores


def test_prediction_examples(predictor):
    """Test predictions on specific examples"""
    print("\n🔮 Testing predictions on specific examples...")
    
    test_cases = [
        {"formula": "Fe3O4", "expected_rl": "excellent", "expected_eab": "excellent"},
        {"formula": "TiO2", "expected_rl": "good", "expected_eab": "good"},
        {"formula": "ZnO", "expected_rl": "poor", "expected_eab": "poor"},
        {"formula": "NiFe2O4", "expected_rl": "excellent", "expected_eab": "excellent"},
        {"formula": "Al2O3", "expected_rl": "poor", "expected_eab": "poor"},
    ]
    
    for case in test_cases:
        result = predictor.predict(case["formula"])
        rl_pred = result['rl_predictions'][0]
        eab_pred = result['eab_predictions'][0]
        rl_conf = max(result['rl_probabilities'][0])
        eab_conf = max(result['eab_probabilities'][0])
        
        rl_match = "✅" if rl_pred == case["expected_rl"] else "❌"
        eab_match = "✅" if eab_pred == case["expected_eab"] else "❌"
        
        print(f"🧪 {case['formula']}:")
        print(f"   RL: {rl_pred} (conf: {rl_conf:.2f}) {rl_match}")
        print(f"   EAB: {eab_pred} (conf: {eab_conf:.2f}) {eab_match}")


def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("🚀 Starting Comprehensive PLS Predictor Test")
    print("=" * 60)
    
    # Generate test data
    formulas, rl_values, eab_values, categories = generate_realistic_test_data(n_samples=150)
    
    # Cross-validation test
    cv_rl_scores, cv_eab_scores = test_cross_validation(formulas, rl_values, eab_values)
    
    # Train/test split evaluation
    print(f"\n🎯 Training and evaluating predictor...")
    predictor = create_predictor(use_matminer=True, n_components=8)
    
    # Extract features
    X = predictor.feature_extractor.extract_features(formulas)
    
    # Prepare targets
    rl_array = np.array(rl_values)
    eab_array = np.array(eab_values)
    rl_classes, eab_classes = predictor._discretize_targets(rl_array, eab_array)
    
    # Split data manually for detailed evaluation
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(formulas))
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    rl_train, rl_test = rl_classes[train_idx], rl_classes[test_idx]
    eab_train, eab_test = eab_classes[train_idx], eab_classes[test_idx]
    
    # Encode and scale
    rl_train_encoded = predictor.rl_label_encoder.fit_transform(rl_train)
    rl_test_encoded = predictor.rl_label_encoder.transform(rl_test)
    eab_train_encoded = predictor.eab_label_encoder.fit_transform(eab_train)
    eab_test_encoded = predictor.eab_label_encoder.transform(eab_test)
    
    X_train_scaled = predictor.scaler.fit_transform(X_train)
    X_test_scaled = predictor.scaler.transform(X_test)
    
    # Train models
    n_components = min(8, X_train_scaled.shape[1], X_train_scaled.shape[0] - 1)
    predictor.rl_model = predictor._create_pls_pipeline(n_components)
    predictor.eab_model = predictor._create_pls_pipeline(n_components)
    
    predictor.rl_model.fit(X_train_scaled, rl_train_encoded)
    predictor.eab_model.fit(X_train_scaled, eab_train_encoded)
    predictor.is_trained = True
    
    # Evaluate
    eval_results = evaluate_predictor(predictor, X_test, rl_test, eab_test, 
                                    rl_test_encoded, eab_test_encoded)
    
    # Plot confusion matrices
    plot_confusion_matrices(predictor, rl_test_encoded, eab_test_encoded,
                           eval_results['rl_pred'], eval_results['eab_pred'])
    
    # Test specific examples
    test_prediction_examples(predictor)
    
    # Summary
    print(f"\n🎉 Test Summary")
    print("=" * 60)
    print(f"📊 Cross-validation Performance:")
    print(f"   RL CV Accuracy: {cv_rl_scores.mean():.3f} ± {cv_rl_scores.std():.3f}")
    print(f"   EAB CV Accuracy: {cv_eab_scores.mean():.3f} ± {cv_eab_scores.std():.3f}")
    
    print(f"\n📊 Test Set Performance:")
    print(f"   RL Accuracy: {eval_results['rl_accuracy']:.3f}")
    print(f"   RL F1 Score: {eval_results['rl_f1']:.3f}")
    print(f"   EAB Accuracy: {eval_results['eab_accuracy']:.3f}")
    print(f"   EAB F1 Score: {eval_results['eab_f1']:.3f}")
    
    print(f"\n📋 Dataset Information:")
    print(f"   Total samples: {len(formulas)}")
    print(f"   Training samples: {len(train_idx)}")
    print(f"   Test samples: {len(test_idx)}")
    print(f"   Features: {X.shape[1]}")
    
    # Performance interpretation
    print(f"\n💡 Performance Interpretation:")
    avg_accuracy = (eval_results['rl_accuracy'] + eval_results['eab_accuracy']) / 2
    if avg_accuracy >= 0.8:
        print("   🎯 Excellent performance! The predictor works very well.")
    elif avg_accuracy >= 0.7:
        print("   ✅ Good performance! The predictor is reliable.")
    elif avg_accuracy >= 0.6:
        print("   ⚠️ Moderate performance. Consider more training data or feature engineering.")
    else:
        print("   ❌ Poor performance. Significant improvements needed.")
    
    # Save test results
    test_results = {
        'cv_rl_mean': cv_rl_scores.mean(),
        'cv_rl_std': cv_rl_scores.std(),
        'cv_eab_mean': cv_eab_scores.mean(),
        'cv_eab_std': cv_eab_scores.std(),
        'test_rl_accuracy': eval_results['rl_accuracy'],
        'test_rl_f1': eval_results['rl_f1'],
        'test_eab_accuracy': eval_results['eab_accuracy'],
        'test_eab_f1': eval_results['eab_f1'],
        'n_samples': len(formulas),
        'n_features': X.shape[1]
    }
    
    import json
    with open('test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n💾 Test results saved to 'test_results.json'")
    
    return predictor, test_results


if __name__ == "__main__":
    try:
        predictor, results = run_comprehensive_test()
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 