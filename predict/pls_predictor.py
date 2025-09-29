"""
PLS Predictor for Molecular Formulas
Uses preprocessed datasets and matminer features for RL and EAB prediction
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from typing import Dict, List, Tuple, Union, Optional
import re
import sys

warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import required libraries
try:
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: sklearn not installed, PLS predictor unavailable")
    SKLEARN_AVAILABLE = False

try:
    from pymatgen.core import Composition
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import (
        Stoichiometry, ElementFraction,
        ElementProperty, ValenceOrbital
    )
    MATMINER_AVAILABLE = True
except ImportError:
    print("Warning: matminer/pymatgen not installed, will use basic features only")
    MATMINER_AVAILABLE = False

# Import local feature enhancer
try:
    from preprocessing.feature_enhancer import FeatureEnhancer
    FEATURE_ENHANCER_AVAILABLE = True
except ImportError:
    print("Warning: local feature enhancer not available")
    FEATURE_ENHANCER_AVAILABLE = False


# Custom PLS Transformer for proper dimensionality handling
from sklearn.base import BaseEstimator, TransformerMixin

class PLSTransformer(BaseEstimator, TransformerMixin):
    """Custom PLS transformer that ensures proper 2D output"""
    
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=n_components)
        
    def fit(self, X, y):
        self.pls.fit(X, y)
        return self
        
    def transform(self, X):
        X_transformed = self.pls.transform(X)
        # Ensure 2D output for LogisticRegression
        if X_transformed.ndim == 3:
            X_transformed = X_transformed.reshape(X_transformed.shape[0], -1)
        elif X_transformed.ndim == 1:
            X_transformed = X_transformed.reshape(-1, 1)
        return X_transformed


class FormulaExtractor:
    """Extract chemical formula from datasets"""
    
    def __init__(self):
        """Initialize formula extractor"""
        pass
    
    def extract_formula_from_features(self, df: pd.DataFrame) -> List[str]:
        """
        Extract chemical formulas from feature columns
        """
        formulas = []
        
        # Element columns (from periodic table)
        element_columns = [
            'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar',
            'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr',
            'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe',
            'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
            'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',
            'Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr',
            'Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og'
        ]
        
        for idx, row in df.iterrows():
            # Extract elements with non-zero fractions
            formula_parts = []
            
            for element in element_columns:
                if element in df.columns:
                    fraction = row[element]
                    if pd.notna(fraction) and fraction > 0:
                        if fraction >= 0.5:
                            count = max(1, round(fraction * 10))  # Scale up for better representation
                            if count == 1:
                                formula_parts.append(element)
                            else:
                                formula_parts.append(f"{element}{count}")
                        elif fraction >= 0.1:
                            formula_parts.append(element)
            
            if formula_parts:
                formula = ''.join(formula_parts)
                formulas.append(formula)
            else:
                # Fallback formula if no elements detected
                formulas.append('C')
        
        return formulas


class PLSPredictor:
    """PLS Predictor for RL and EAB values using preprocessed datasets"""
    
    def __init__(self, datasets_dir: str = "./datasets", use_all_features: bool = True, n_components: int = 10):
        """Initialize PLS predictor"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for PLS predictor")
        
        self.datasets_dir = datasets_dir
        self.use_all_features = use_all_features
        self.n_components = n_components
        
        # Feature enhancer for generating features from formulas (when predicting new formulas)
        if FEATURE_ENHANCER_AVAILABLE:
            self.feature_enhancer = FeatureEnhancer()
        else:
            self.feature_enhancer = None
            
        self.formula_extractor = FormulaExtractor()
        
        # Models for RL and EAB
        self.rl_model = None
        self.eab_model = None
        
        # Scalers and encoders
        self.scaler = StandardScaler()
        self.rl_label_encoder = LabelEncoder()
        self.eab_label_encoder = LabelEncoder()
        
        # Training data for future predictions
        self.training_features = None
        self.feature_columns = None
        
        # Training status
        self.is_trained = False
        
    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load RL and EAB datasets"""
        print("📂 Loading datasets...")
        
        # Load RL dataset
        rl_train_path = os.path.join(self.datasets_dir, "rl_class_train.csv")
        rl_test_path = os.path.join(self.datasets_dir, "rl_class_test.csv")
        
        if not os.path.exists(rl_train_path):
            raise FileNotFoundError(f"RL training dataset not found: {rl_train_path}")
        if not os.path.exists(rl_test_path):
            raise FileNotFoundError(f"RL test dataset not found: {rl_test_path}")
            
        rl_train = pd.read_csv(rl_train_path)
        rl_test = pd.read_csv(rl_test_path)
        
        # Load EAB dataset
        eab_train_path = os.path.join(self.datasets_dir, "eab_class_train.csv")
        eab_test_path = os.path.join(self.datasets_dir, "eab_class_test.csv")
        
        if not os.path.exists(eab_train_path):
            raise FileNotFoundError(f"EAB training dataset not found: {eab_train_path}")
        if not os.path.exists(eab_test_path):
            raise FileNotFoundError(f"EAB test dataset not found: {eab_test_path}")
            
        eab_train = pd.read_csv(eab_train_path)
        eab_test = pd.read_csv(eab_test_path)
        
        print(f"✅ Loaded RL dataset: train={rl_train.shape}, test={rl_test.shape}")
        print(f"✅ Loaded EAB dataset: train={eab_train.shape}, test={eab_test.shape}")
        
        return (rl_train, rl_test), (eab_train, eab_test)
    
    def prepare_features(self, df: pd.DataFrame, use_basic_only: bool = False) -> pd.DataFrame:
        """
        Prepare features from dataset
        """
        print("🔧 Preparing features...")
        
        # Get all columns except targets and metadata
        exclude_cols = ['target', 'rl_min', 'eab', 'thickness']
        
        if use_basic_only:
            # Only use basic features
            feature_cols = ['is_heterostructure', 'is_supported']
        else:
            # Use all available features in the dataset
            feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Ensure the basic features exist
        for col in ['is_heterostructure', 'is_supported']:
            if col not in feature_cols and col in df.columns:
                feature_cols.append(col)
        
        # Extract features
        feature_df = df[feature_cols].copy()
        
        # Handle missing values
        feature_df = feature_df.fillna(0)
        
        print(f"✅ Prepared {feature_df.shape[1]} features for {feature_df.shape[0]} samples")
        
        return feature_df
    
    def train(self, use_grid_search: bool = False, use_basic_only: bool = False) -> Dict:
        """Train PLS models using datasets"""
        print("🚀 Starting PLS model training with datasets...")
        
        # Load datasets
        (rl_train, rl_test), (eab_train, eab_test) = self.load_datasets()
        
        # Prepare RL features and targets
        print("\n📊 Preparing RL dataset...")
        X_rl_train = self.prepare_features(rl_train, use_basic_only)
        X_rl_test = self.prepare_features(rl_test, use_basic_only)
        y_rl_train = rl_train['target'].values
        y_rl_test = rl_test['target'].values
        
        # Prepare EAB features and targets
        print("\n📊 Preparing EAB dataset...")
        X_eab_train = self.prepare_features(eab_train, use_basic_only)
        X_eab_test = self.prepare_features(eab_test, use_basic_only)
        y_eab_train = eab_train['target'].values
        y_eab_test = eab_test['target'].values
        
        # Store feature columns for future predictions
        self.feature_columns = X_rl_train.columns.tolist()
        
        # Scale features
        print("\n⚖️ Scaling features...")
        X_rl_train_scaled = self.scaler.fit_transform(X_rl_train)
        X_rl_test_scaled = self.scaler.transform(X_rl_test)
        
        # For EAB, we need a separate scaler or use the same one
        # Using the same scaler assumes similar feature distributions
        X_eab_train_scaled = self.scaler.transform(X_eab_train)
        X_eab_test_scaled = self.scaler.transform(X_eab_test)
        
        # Encode labels
        y_rl_train_encoded = self.rl_label_encoder.fit_transform(y_rl_train)
        y_rl_test_encoded = self.rl_label_encoder.transform(y_rl_test)
        
        y_eab_train_encoded = self.eab_label_encoder.fit_transform(y_eab_train)
        y_eab_test_encoded = self.eab_label_encoder.transform(y_eab_test)
        
        # Determine optimal n_components
        n_components = min(self.n_components, X_rl_train_scaled.shape[1], X_rl_train_scaled.shape[0] - 1)
        
        # Train RL model
        print("🎯 Training RL model...")
        self.rl_model = self._create_pls_pipeline(n_components)
        
        if use_grid_search:
            param_grid = {
                'pls__n_components': [min(5, n_components), min(10, n_components), min(20, n_components)],
                'classifier__C': [0.1, 1.0, 10.0]
            }
            rl_grid = GridSearchCV(self.rl_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            rl_grid.fit(X_rl_train_scaled, y_rl_train_encoded)
            self.rl_model = rl_grid.best_estimator_
        else:
            self.rl_model.fit(X_rl_train_scaled, y_rl_train_encoded)
        
        # Train EAB model
        print("🎯 Training EAB model...")
        self.eab_model = self._create_pls_pipeline(n_components)
        
        if use_grid_search:
            eab_grid = GridSearchCV(self.eab_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            eab_grid.fit(X_eab_train_scaled, y_eab_train_encoded)
            self.eab_model = eab_grid.best_estimator_
        else:
            self.eab_model.fit(X_eab_train_scaled, y_eab_train_encoded)
        
        self.is_trained = True
        
        # Store training data for future predictions
        self.training_features = X_rl_train.copy()
        
        # Evaluate models
        print("\n📊 Evaluating models...")
        rl_pred = self.rl_model.predict(X_rl_test_scaled)
        eab_pred = self.eab_model.predict(X_eab_test_scaled)
        
        results = {
            'rl_accuracy': accuracy_score(y_rl_test_encoded, rl_pred),
            'rl_f1': f1_score(y_rl_test_encoded, rl_pred, average='weighted'),
            'eab_accuracy': accuracy_score(y_eab_test_encoded, eab_pred),
            'eab_f1': f1_score(y_eab_test_encoded, eab_pred, average='weighted'),
            'rl_train_size': len(y_rl_train),
            'rl_test_size': len(y_rl_test),
            'eab_train_size': len(y_eab_train),
            'eab_test_size': len(y_eab_test),
            'n_features': X_rl_train.shape[1],
            'n_components': n_components
        }
        
        print("✅ Training completed!")
        print(f"📊 RL Model - Accuracy: {results['rl_accuracy']:.3f}, F1: {results['rl_f1']:.3f}")
        print(f"📊 EAB Model - Accuracy: {results['eab_accuracy']:.3f}, F1: {results['eab_f1']:.3f}")
        
        return results
    
    def _create_pls_pipeline(self, n_components: int) -> Pipeline:
        """Create PLS pipeline with proper transformer"""
        pipeline = Pipeline([
            ('pls', PLSTransformer(n_components=n_components)),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        return pipeline
    
    def predict_from_formula(self, formula: str) -> Dict:
        """
        Predict RL and EAB values from a chemical formula
        
        Args:
            formula: Chemical formula string (e.g., "Ti3C2", "Fe3O4")
            
        Returns:
            Dictionary with RL and EAB predictions and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        print(f"🔮 Predicting for formula: {formula}")
        
        # Create features for the input formula
        feature_dict = {col: 0.0 for col in self.feature_columns}
        
        # Set basic features to default values
        feature_dict['is_heterostructure'] = 0
        feature_dict['is_supported'] = 0
        
        # Generate matminer features if available
        if self.feature_enhancer:
            try:
                print("🧪 Extracting matminer features...")
                temp_df = pd.DataFrame({'formula': [formula]})
                enhanced_df = self.feature_enhancer.enhance_features(temp_df)
                
                # Update feature dictionary with matminer features
                for col in enhanced_df.columns:
                    if col != 'formula' and col in feature_dict:
                        feature_dict[col] = enhanced_df[col].iloc[0] if not pd.isna(enhanced_df[col].iloc[0]) else 0.0
                        
            except Exception as e:
                print(f"⚠️ Matminer feature extraction failed: {e}")
                print("   Using basic features only")
        
        # Create feature dataframe
        feature_df = pd.DataFrame([feature_dict])
        
        # Ensure all columns are present and in correct order
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        
        feature_df = feature_df[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(feature_df)
        
        # Make predictions
        rl_pred_encoded = self.rl_model.predict(X_scaled)[0]
        eab_pred_encoded = self.eab_model.predict(X_scaled)[0]
        
        # Decode predictions
        rl_pred = self.rl_label_encoder.inverse_transform([rl_pred_encoded])[0]
        eab_pred = self.eab_label_encoder.inverse_transform([eab_pred_encoded])[0]
        
        # Get prediction probabilities
        rl_proba = self.rl_model.predict_proba(X_scaled)[0]
        eab_proba = self.eab_model.predict_proba(X_scaled)[0]
        
        results = {
            'formula': formula,
            'rl_prediction': rl_pred,
            'eab_prediction': eab_pred,
            'rl_probabilities': dict(zip(self.rl_label_encoder.classes_, rl_proba)),
            'eab_probabilities': dict(zip(self.eab_label_encoder.classes_, eab_proba)),
            'rl_confidence': float(max(rl_proba)),
            'eab_confidence': float(max(eab_proba))
        }
        
        return results
    
    def save_model(self, filepath: str = "trained_pls_model.pkl"):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'rl_model': self.rl_model,
            'eab_model': self.eab_model,
            'scaler': self.scaler,
            'rl_label_encoder': self.rl_label_encoder,
            'eab_label_encoder': self.eab_label_encoder,
            'feature_enhancer': self.feature_enhancer,
            'n_components': self.n_components,
            'use_all_features': self.use_all_features,
            'feature_columns': self.feature_columns,
            'training_features': self.training_features
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str = "trained_pls_model.pkl"):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.rl_model = model_data['rl_model']
        self.eab_model = model_data['eab_model']
        self.scaler = model_data['scaler']
        self.rl_label_encoder = model_data['rl_label_encoder']
        self.eab_label_encoder = model_data['eab_label_encoder']
        self.feature_enhancer = model_data.get('feature_enhancer')
        self.n_components = model_data['n_components']
        self.use_all_features = model_data.get('use_all_features', True)
        self.feature_columns = model_data.get('feature_columns', [])
        self.training_features = model_data.get('training_features')
        
        self.is_trained = True
        print(f"✅ Model loaded from {filepath}")


# Convenience functions
def train_and_save_model(datasets_dir: str = "./datasets", 
                        model_path: str = "trained_pls_model.pkl",
                        use_basic_only: bool = False,
                        use_grid_search: bool = False) -> Dict:
    """
    Train PLS model and save it
    
    Args:
        datasets_dir: Directory containing the datasets
        model_path: Path to save the trained model
        use_basic_only: Whether to use only basic features (is_heterostructure, is_supported)
        use_grid_search: Whether to use grid search for hyperparameter tuning
        
    Returns:
        Training results dictionary
    """
    print("🎯 Training and saving PLS model...")
    
    predictor = PLSPredictor(datasets_dir=datasets_dir, use_all_features=not use_basic_only)
    results = predictor.train(use_grid_search=use_grid_search, use_basic_only=use_basic_only)
    predictor.save_model(model_path)
    
    return results


def predict_formula_properties(formula: str, model_path: str = "trained_pls_model.pkl") -> Dict:
    """
    Predict EAB and RL values for a given chemical formula
    
    Args:
        formula: Chemical formula (e.g., "Ti3C2", "Fe3O4")
        model_path: Path to the trained model file
        
    Returns:
        Dictionary with EAB and RL predictions
    """
    predictor = PLSPredictor()
    predictor.load_model(model_path)
    return predictor.predict_from_formula(formula)


if __name__ == "__main__":
    print("🧪 PLS Predictor with Datasets")
    print("=" * 50)
    
    try:
        # Train and save model
        print("🎯 Training model...")
        results = train_and_save_model(
            datasets_dir="./datasets",
            use_basic_only=False,  # Use all features in datasets
            use_grid_search=False
        )
        
        print(f"\n📊 Training Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        
        # Test prediction
        print("\n🔮 Testing prediction...")
        test_formulas = ["Ti3C2", "Fe3O4", "C"]
        
        for formula in test_formulas:
            try:
                prediction = predict_formula_properties(formula)
                print(f"\n📝 {formula}:")
                print(f"  RL: {prediction['rl_prediction']} (confidence: {prediction['rl_confidence']:.3f})")
                print(f"  EAB: {prediction['eab_prediction']} (confidence: {prediction['eab_confidence']:.3f})")
            except Exception as e:
                print(f"  ❌ Prediction failed for {formula}: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("  1. datasets/ directory exists with train/test CSV files")
        print("  2. Required packages are installed (sklearn, pandas, numpy)")
        print("  3. Feature enhancer is available") 