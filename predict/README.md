# Material Property Prediction System

This is a machine learning system for predicting microwave absorption properties (RL and EAB values) of materials based on their chemical formulas.

## Features

✅ **Data Preprocessing**: Automatically filters records with RL or EAB values and molecular formulas  
✅ **Feature Engineering**: Parses chemical formula structures, distinguishes heterostructures and supported structures  
✅ **Feature Enhancement**: Uses matminer to add material physicochemical features  
✅ **Multi-Algorithm Support**: PLS, Random Forest, DNN, MLP algorithms  
✅ **Automatic Evaluation**: Generates detailed performance comparison reports and visualization charts  

## System Architecture

```
predict/
├── main.py                 # Main program entry point
├── core/                   # Core algorithms
│   └── algorithms/         # Algorithm implementations
│       ├── __init__.py
│       ├── base_algorithm.py   # Base algorithm class
│       ├── pls_algorithm.py    # PLS algorithm
│       ├── rf_algorithm.py     # Random Forest algorithm
│       ├── dnn_algorithm.py    # Deep Neural Network algorithm
│       └── mlp_algorithm.py    # Multi-layer Perceptron algorithm
├── preprocessing/          # Data preprocessing modules
│   ├── __init__.py
│   ├── data_processor.py       # Data preprocessing module
│   ├── data_quality_filter.py  # Data quality filter
│   ├── data_balancer.py        # Data balancer
│   ├── data_splitter.py        # Dataset splitting module
│   ├── feature_enhancer.py     # Feature enhancement module
│   └── feature_optimizer.py    # Feature optimizer
├── evaluation/             # Evaluation modules
│   ├── __init__.py
│   └── evaluator.py           # Result evaluation module
├── utils/                  # Utility modules
│   └── __init__.py
├── scripts/                # Utility scripts
│   ├── __init__.py
│   ├── demo_data_filtering.py
│   ├── test_pls_features.py
│   ├── generate_eab_balanced_ultra_quality.py
│   └── generate_ultra_high_quality_rl_balanced.py
└── README.md              # Documentation
```

## Installation

### Basic Dependencies (Required)
```bash
pip install pandas numpy scikit-learn
```

### Deep Learning Support (Optional)
```bash
pip install tensorflow
```

### Material Feature Enhancement (Optional)
```bash
pip install matminer pymatgen
```

### Visualization Support (Optional)
```bash
pip install matplotlib seaborn
```

## Usage

### Basic Usage

```bash
# Run with default settings
python main.py

# Specify data file
python main.py --data ../json/all.json

# Specify output directory
python main.py --output ./my_results
```

### Advanced Options

```bash
# Fast mode (fewer parameters, suitable for testing)
python main.py --fast

# Use grid search for parameter optimization (time-consuming but may get better results)
python main.py --grid-search

# Disable matminer feature enhancement (if matminer is not installed)
python main.py --no-matminer

# Specify cache directory
python main.py --cache ./my_cache

# Force rebuild dataset (ignore cache)
python main.py --force-rebuild

# Combine options
python main.py --fast --no-matminer --output ./quick_test --cache ./temp_cache
```

### Command Line Arguments

- `--data`: JSON data file path, default: `../json/all.json`
- `--output`: Result output directory, default: `./results`
- `--cache`: Dataset cache directory, default: `./datasets`
- `--no-matminer`: Disable matminer feature enhancement
- `--grid-search`: Use grid search for algorithm parameter optimization
- `--fast`: Fast mode with fewer parameters and training epochs
- `--force-rebuild`: Force rebuild dataset (ignore cache)

## Output Results

After completion, the following files will be generated in the output directory:

### Summary Files
- `summary_report.json`: Detailed comparison report
- `evaluation_results.json`: Detailed results for all algorithms
- `{task_name}_summary.csv`: Summary table for each task

### Visualization Charts
- `{task_name}_comparison.png`: Algorithm performance comparison chart

### Cache Files (in cache directory)
- `{task_name}_train.csv`: Training set (including features and labels)
- `{task_name}_test.csv`: Test set (including features and labels)

### Console Output
The program displays:
- Cache detection results
- Data preprocessing progress
- Feature extraction results
- Algorithm training process
- Final performance ranking

## Caching Mechanism

### Automatic Caching
The system automatically saves processed datasets to the cache directory:
- First run: Automatically saves after completing data preprocessing and feature extraction
- Subsequent runs: Loads valid cache directly, greatly saving time

### Cache Benefits
- **Significant speedup**: Skips time-consuming matminer feature extraction (usually takes minutes to tens of minutes)
- **Consistency**: Ensures multiple runs use the same dataset
- **Debugging friendly**: Focus on algorithm tuning without repeating data processing

### Cache Management
```bash
# View cache contents
ls ./datasets

# Force rebuild cache
python main.py --force-rebuild

# Use different cache directory
python main.py --cache ./experiment_cache
```

## Data Format Requirements

Input JSON file should contain the following structure:

```json
[
  {
    "doi": "Literature identifier",
    "content": [
      {
        "record_designation": "Sample ID",
        "general_properties": {
          "chemical_formula": "Ti3C2Tx-NiCo2S4"
        },
        "microwave_absorption_properties": {
          "rl_min": {"value": -45.15, "unit": "dB"},
          "eab": {"value": 3.34, "unit": "GHz"}
        }
      }
    ]
  }
]
```

## Classification Tasks

The system automatically creates the following classification tasks:

1. **RL Classification**: 
   - Class 0: RL ≤ -50 dB (better performance)
   - Class 1: RL > -50 dB (worse performance)

2. **EAB Classification**:
   - Class 0: EAB ≤ 4 GHz
   - Class 1: EAB > 4 GHz (better performance)

## Feature Engineering

### Chemical Formula Parsing
- Automatically identifies heterostructures (containing `/` symbol)
- Automatically identifies supported structures (containing `@` symbol)
- Handles composite materials (containing `-`, `_`, `·` separators)

### Material Feature Enhancement (matminer)
- Stoichiometric features
- Element fraction features
- Element property features (magpie dataset)
- Valence electron orbital features

## Algorithm Description

### PLS (Partial Least Squares)
- Suitable for high-dimensional data
- Can handle cases where feature dimensions exceed sample size
- Supports parameter optimization

### Random Forest
- Ensemble learning algorithm
- Provides feature importance ranking
- Robust to missing values and outliers

### MLP (Multi-layer Perceptron)
- sklearn implementation of neural network
- Supports early stopping and learning rate adjustment
- Provides training process monitoring

### DNN (Deep Neural Network)
- TensorFlow/Keras implementation
- Supports multi-layer network architecture
- Includes Dropout and Batch Normalization

## Performance Evaluation Metrics

- **Accuracy**: Proportion of correct predictions
- **Precision**: Proportion of correct predictions among positive predictions
- **Recall**: Proportion of actual positive samples identified
- **F1-Score**: Harmonic mean of precision and recall
- **Training Time**: Model training duration
- **Prediction Time**: Model prediction duration

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'matminer'**
   - Solution: Use `--no-matminer` parameter or install matminer

2. **ImportError: No module named 'tensorflow'**
   - Solution: Install tensorflow or ignore DNN algorithm

3. **Training failure due to insufficient data**
   - Solution: Check data quality, ensure sufficient valid samples

4. **Out of memory**
   - Solution: Use `--fast` mode or reduce sample size

### Debug Mode

Set debug information in code:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Extension Development

### Adding New Algorithms

1. Create new algorithm file in `core/algorithms/` directory
2. Inherit from `BaseAlgorithm` class
3. Implement required abstract methods
4. Import in `core/algorithms/__init__.py`
5. Add in `setup_algorithms()` function in `main.py`

### Custom Features

1. Modify feature extraction logic in `preprocessing/feature_enhancer.py`
2. Add new feature generation functions
3. Update feature list

## License

This project is for academic research use only.

## Contact

For questions, please create a GitHub Issue or contact the developer. 