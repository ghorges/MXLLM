"""
Main Program
Integrates all modules to run the complete material property prediction workflow
"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.data_processor import DataProcessor
from preprocessing.feature_enhancer import FeatureEnhancer, enhance_dataset_features
from preprocessing.data_splitter import DataSplitter
from evaluation.evaluator import Evaluator

try:
    from core.algorithms import PLSAlgorithm, RandomForestAlgorithm, DNNAlgorithm, MLPAlgorithm
    ALGORITHMS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Algorithm module import failed: {e}")
    ALGORITHMS_AVAILABLE = False


def setup_algorithms(use_grid_search: bool = False, fast_mode: bool = False):
    """Setup algorithm list"""
    algorithms = []
    
    try:
        from core.algorithms.pls_algorithm import PLSAlgorithm
        algorithms.append(PLSAlgorithm(use_grid_search=False))
        print("✅ PLS algorithm loaded")
    except ImportError as e:
        print(f"❌ PLS algorithm loading failed: {e}")
    
    try:
        from core.algorithms.rf_algorithm import RandomForestAlgorithm
        algorithms.append(RandomForestAlgorithm(
            n_estimators=100,
            use_grid_search=False
        ))
        print("✅ Random Forest algorithm loaded")
    except ImportError as e:
        print(f"❌ Random Forest algorithm loading failed: {e}")
    
    try:
        from core.algorithms.mlp_algorithm import MLPAlgorithm
        algorithms.append(MLPAlgorithm(
            max_iter=1500 if fast_mode else 3000,
            use_grid_search=use_grid_search
        ))
        print("✅ MLP algorithm loaded")
    except ImportError as e:
        print(f"❌ MLP algorithm loading failed: {e}")
    
    try:
        from core.algorithms.dnn_algorithm import DNNAlgorithm
        algorithms.append(DNNAlgorithm(
            epochs=200 if fast_mode else 500,
            use_grid_search=use_grid_search
        ))
        print("✅ DNN algorithm loaded")
    except ImportError as e:
        print(f"❌ DNN algorithm loading failed: {e}")
    
    return algorithms


def check_cache_exists(cache_dir: str, target_name: str) -> bool:
    """Check if cache files exist"""
    train_file = os.path.join(cache_dir, f"{target_name}_train.csv")
    test_file = os.path.join(cache_dir, f"{target_name}_test.csv")
    
    exists = os.path.exists(train_file) and os.path.exists(test_file)
    if exists:
        print(f"✅ Cache found for {target_name}")
        print(f"   Training set: {train_file}")
        print(f"   Test set: {test_file}")
    else:
        print(f"❌ No cache found for {target_name}")
    
    return exists


def load_cached_data(cache_dir: str, target_name: str):
    """Load cached data"""
    import pandas as pd
    
    train_file = os.path.join(cache_dir, f"{target_name}_train.csv")
    test_file = os.path.join(cache_dir, f"{target_name}_test.csv")
    
    print(f"📂 Loading cached data for {target_name}...")
    
    train_df = pd.read_csv(train_file)
    X_train = train_df.drop(['target'], axis=1)
    y_train = train_df['target']
    
    test_df = pd.read_csv(test_file)
    X_test = test_df.drop(['target'], axis=1)
    y_test = test_df['target']
    
    print(f"   ✅ Training set: {X_train.shape}")
    print(f"   ✅ Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def save_data_to_cache(cache_dir: str, target_name: str, X_train, X_test, y_train, y_test):
    """Save data to cache"""
    import pandas as pd
    
    os.makedirs(cache_dir, exist_ok=True)
    
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_file = os.path.join(cache_dir, f"{target_name}_train.csv")
    train_df.to_csv(train_file, index=False)
    
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_file = os.path.join(cache_dir, f"{target_name}_test.csv")
    test_df.to_csv(test_file, index=False)
    
    print(f"💾 Cached data saved for {target_name}")
    print(f"   Training set: {train_file}")
    print(f"   Test set: {test_file}")


def process_target(json_file: str, target_name: str, algorithms: list, 
                  output_dir: str, cache_dir: str, use_matminer: bool = True, 
                  force_rebuild: bool = False):
    """Process a single prediction target"""
    print(f"\n{'='*50}")
    print(f"🎯 Processing target: {target_name}")
    print(f"{'='*50}")
    
    use_cache = False
    if not force_rebuild and check_cache_exists(cache_dir, target_name):
        try:
            X_train, X_test, y_train, y_test = load_cached_data(cache_dir, target_name)
            use_cache = True
            print(f"✅ Using cached data for {target_name}")
        except Exception as e:
            print(f"⚠️ Cache loading failed: {e}")
            print(f"   Rebuilding dataset...")
            use_cache = False
    
    if not use_cache:
        print(f"🔄 Building dataset for {target_name}...")
        
        print("1️⃣ Data preprocessing...")
        processor = DataProcessor(json_file)
        df = processor.process_data()
        print(f"   ✅ Processed {len(df)} records")
        
        print("2️⃣ Feature enhancement...")
        try:
            if use_matminer:
                enhancer = FeatureEnhancer()
                df_enhanced = enhance_dataset_features(df, enhancer)
                print(f"   ✅ Enhanced to {df_enhanced.shape[1]} features")
            else:
                print("   ⚠️ Skipping matminer features")
                df_enhanced = df
        except Exception as e:
            print(f"   ⚠️ Feature enhancement failed: {e}")
            print(f"   📊 Using basic features only")
            df_enhanced = df
        
        print("3️⃣ Data splitting...")
        splitter = DataSplitter()
        X_train, X_test, y_train, y_test = splitter.split_data(df_enhanced, target_name)
        print(f"   ✅ Training set: {X_train.shape}")
        print(f"   ✅ Test set: {X_test.shape}")
        
        save_data_to_cache(cache_dir, target_name, X_train, X_test, y_train, y_test)
    
    print("4️⃣ Model training and evaluation...")
    evaluator = Evaluator()
    results = evaluator.evaluate_algorithms(algorithms, X_train, X_test, y_train, y_test, target_name)
    
    print("5️⃣ Saving results...")
    os.makedirs(output_dir, exist_ok=True)
    evaluator.save_results(results, target_name, output_dir)
    
    print(f"✅ {target_name} processing completed!")
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Material Property Prediction System')
    parser.add_argument('--data', default='../json/all.json', help='JSON data file path')
    parser.add_argument('--output', default='./results', help='Output directory')
    parser.add_argument('--cache', default='./datasets', help='Cache directory')
    parser.add_argument('--no-matminer', action='store_true', help='Disable matminer features')
    parser.add_argument('--grid-search', action='store_true', help='Use grid search for optimization')
    parser.add_argument('--fast', action='store_true', help='Fast mode with fewer parameters')
    parser.add_argument('--force-rebuild', action='store_true', help='Force rebuild dataset (ignore cache)')
    
    args = parser.parse_args()
    
    print("🚀 Material Property Prediction System")
    print("=" * 50)
    print(f"📁 Data file: {args.data}")
    print(f"📁 Output directory: {args.output}")
    print(f"📁 Cache directory: {args.cache}")
    print(f"🧪 Use matminer: {not args.no_matminer}")
    print(f"🔍 Grid search: {args.grid_search}")
    print(f"⚡ Fast mode: {args.fast}")
    print(f"🔄 Force rebuild: {args.force_rebuild}")
    
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        return
    
    print("\n📦 Setting up algorithms...")
    algorithms = setup_algorithms(use_grid_search=args.grid_search, fast_mode=args.fast)
    
    if not algorithms:
        print("❌ No algorithms available!")
        return
    
    print(f"✅ {len(algorithms)} algorithms loaded")
    
    all_results = {}
    targets = ['rl_class', 'eab_class']
    
    for target in targets:
        try:
            results = process_target(
                args.data, target, algorithms, args.output, args.cache,
                use_matminer=not args.no_matminer,
                force_rebuild=args.force_rebuild
            )
            all_results[target] = results
        except Exception as e:
            print(f"❌ Failed to process {target}: {e}")
            continue
    
    if all_results:
        print(f"\n{'='*50}")
        print("📊 SUMMARY REPORT")
        print(f"{'='*50}")
        
        evaluator = Evaluator()
        evaluator.generate_summary_report(all_results, args.output)
        
        print("🎉 All tasks completed!")
        print(f"📁 Results saved to: {args.output}")
    else:
        print("❌ No results generated!")


if __name__ == "__main__":
    main() 