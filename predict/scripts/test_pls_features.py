"""
Test PLSDA classification performance with different feature counts
Uses PLSDA's VIP (Variable Importance in Projection) for feature selection
Feature range: 3-50, step size 5
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
from preprocessing.data_splitter import DataSplitter
import time


class PLSDAClassifier:
    """PLSDA classifier with VIP feature importance calculation"""
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=n_components)
        self.label_encoder = LabelEncoder()
        self.vip_scores = None
        
    def fit(self, X, y):
        """Train PLSDA model"""
        y_encoded = self.label_encoder.fit_transform(y)
        self.pls.fit(X, y_encoded)
        
        try:
            self.vip_scores = self._calculate_vip(X, y_encoded)
        except Exception as e:
            print(f"   ⚠️ VIP calculation error: {e}")
            print(f"   📊 Using default VIP scores (all 1s)")
            self.vip_scores = np.ones(X.shape[1])
        
        return self
    
    def predict(self, X):
        """Predict"""
        y_pred_continuous = self.pls.predict(X)
        y_pred_rounded = np.round(y_pred_continuous.flatten()).astype(int)
        y_pred_clipped = np.clip(y_pred_rounded, 0, len(self.label_encoder.classes_) - 1)
        return self.label_encoder.inverse_transform(y_pred_clipped)
    
    def _calculate_vip(self, X, y):
        """Calculate VIP (Variable Importance in Projection) scores"""
        W = self.pls.x_weights_
        Q = self.pls.y_loadings_
        
        ss_y = []
        for i in range(self.n_components):
            if Q.ndim == 1:
                ss_y_i = Q[i] ** 2 if i < len(Q) else 0
            else:
                ss_y_i = np.sum(Q[:, i] ** 2)
            ss_y.append(ss_y_i)
        
        total_ss_y = sum(ss_y)
        
        if total_ss_y == 0:
            return np.ones(X.shape[1])
        
        p = X.shape[1]
        vip_scores = np.zeros(p)
        
        for j in range(p):
            numerator = 0
            for i in range(self.n_components):
                numerator += (W[j, i] ** 2) * ss_y[i]
            
            vip_scores[j] = np.sqrt(p * numerator / total_ss_y)
        
        return vip_scores


def load_dataset_with_cache(task_name, cache_dir="./datasets"):
    """Load dataset from cache"""
    train_file = os.path.join(cache_dir, f"{task_name}_train.csv")
    test_file = os.path.join(cache_dir, f"{task_name}_test.csv")
    
    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        raise FileNotFoundError(f"Cache files not found for {task_name}")
    
    print(f"📂 Loading {task_name} dataset from cache...")
    
    train_df = pd.read_csv(train_file)
    X_train = train_df.drop(['target'], axis=1)
    y_train = train_df['target']
    
    test_df = pd.read_csv(test_file)
    X_test = test_df.drop(['target'], axis=1)
    y_test = test_df['target']
    
    print(f"   ✅ Training set: {X_train.shape}")
    print(f"   ✅ Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def test_pls_with_feature_selection(X_train, X_test, y_train, y_test, task_name, 
                                   feature_range=range(3, 51, 5)):
    """Test PLS performance with different feature counts"""
    print(f"\n🧪 Testing PLS feature selection for {task_name}")
    print(f"Feature range: {min(feature_range)} to {max(feature_range)} (step {feature_range.step})")
    print("-" * 60)
    
    results = []
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    initial_plsda = PLSDAClassifier(n_components=min(10, X_train_scaled.shape[1]))
    initial_plsda.fit(X_train_scaled, y_train)
    vip_scores = initial_plsda.vip_scores
    
    feature_importance = list(zip(range(len(vip_scores)), vip_scores))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for n_features in feature_range:
        if n_features > X_train_scaled.shape[1]:
            print(f"   ⚠️ Skipping {n_features} features (max available: {X_train_scaled.shape[1]})")
            continue
            
        start_time = time.time()
        
        selected_indices = [idx for idx, _ in feature_importance[:n_features]]
        X_train_selected = X_train_scaled[:, selected_indices]
        X_test_selected = X_test_scaled[:, selected_indices]
        
        n_components = min(n_features - 1, 10)
        
        plsda = PLSDAClassifier(n_components=n_components)
        plsda.fit(X_train_selected, y_train)
        y_pred = plsda.predict(X_test_selected)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        training_time = time.time() - start_time
        
        results.append({
            'n_features': n_features,
            'n_components': n_components,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': training_time
        })
        
        print(f"   📊 {n_features:2d} features | "
              f"Acc: {accuracy:.3f} | "
              f"F1: {f1:.3f} | "
              f"Time: {training_time:.2f}s")
    
    return results


def save_results(results, task_name, output_dir="./pls_feature_test_results"):
    """Save test results"""
    os.makedirs(output_dir, exist_ok=True)
    
    df_results = pd.DataFrame(results)
    
    csv_file = os.path.join(output_dir, f"{task_name}_pls_feature_test.csv")
    df_results.to_csv(csv_file, index=False)
    
    best_result = max(results, key=lambda x: x['f1_score'])
    
    print(f"\n📈 Best result for {task_name}:")
    print(f"   🎯 Features: {best_result['n_features']}")
    print(f"   📊 F1 Score: {best_result['f1_score']:.4f}")
    print(f"   🎯 Accuracy: {best_result['accuracy']:.4f}")
    print(f"   💾 Results saved: {csv_file}")
    
    return best_result


def main():
    """Main function"""
    cache_dir = "./datasets"
    output_dir = "./pls_feature_test_results"
    
    tasks = ['rl_class', 'eab_class']
    
    print("🚀 PLS Feature Selection Test")
    print("=" * 50)
    
    all_best_results = {}
    
    for task in tasks:
        try:
            X_train, X_test, y_train, y_test = load_dataset_with_cache(task, cache_dir)
            
            results = test_pls_with_feature_selection(
                X_train, X_test, y_train, y_test, task,
                feature_range=range(3, 51, 5)
            )
            
            best_result = save_results(results, task, output_dir)
            all_best_results[task] = best_result
            
        except Exception as e:
            print(f"❌ Error processing {task}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("📊 SUMMARY")
    print(f"{'='*50}")
    
    for task, result in all_best_results.items():
        print(f"{task:15s} | Best: {result['n_features']:2d} features | F1: {result['f1_score']:.4f}")
    
    print("\n🎉 PLS feature selection test completed!")


if __name__ == "__main__":
    main() 