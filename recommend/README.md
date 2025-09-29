# MXLLM Material Recommendation System

An intelligent MXLLM electromagnetic wave absorption material recommendation system based on RAG (Retrieval-Augmented Generation) and AI prediction technologies.

## Features

### Core Functions
- **Smart Chat**: Natural language conversation with OpenAI GPT models (optional)
- **RAG Recommendations**: Literature retrieval and recommendations using local vector database
- **AI Performance Prediction**: Machine learning-based material performance prediction
- **Three-Type Recommendations**:
  - 🧪 Chemical Formula Recommendations
  - ⚗️ Synthesis Process Recommendations  
  - 🔬 Testing Procedure Recommendations

### System Characteristics
- **Fully Offline**: Core functions work without internet connection
- **Local Database**: ChromaDB local vector database
- **Intelligent Prediction**: Integrated predict module for performance prediction
- **Semantic Search**: SentenceTransformer model runs locally

## Quick Start

### 1. Install Dependencies
```bash
cd recommend
pip install -r requirements.txt
```

### 2. Launch Application
```bash
python main.py
```

### 3. Optional: Configure AI Chat
1. Click "🔑 Configure AI Service" button
2. Enter your OpenAI API key
3. System will automatically validate key

**Note**: Recommendation and prediction functions work fully without AI chat configuration.

## Architecture

### Core Modules
```
recommend/
├── core/                    # Core functionality modules
│   ├── data_loader.py      # Data loader
│   ├── rag_system.py       # RAG retrieval system
│   ├── llm_handler.py      # LLM handler (optional)
│   └── material_validator.py # Material validation and prediction
├── ui/                     # User interface modules
│   ├── main_window.py      # Main window
│   └── modern_window.py    # Modern UI window
├── data/                   # Data directory (auto-created)
│   └── vectordb/          # Vector database
├── logs/                   # Log directory (auto-created)
└── main.py                # Main startup script
```

## Enhanced Features

### Smart Material Validation
The system automatically validates recommended chemical formulas with two approaches:

1. **Database Search Priority**
   - Search experimental data in existing literature
   - Provide real performance parameters and DOI sources
   - Display synthesis methods and testing procedures

2. **AI Prediction Supplement**
   - Automatically call PLS prediction model when no database data exists
   - Predict EAB (Effective Absorption Bandwidth) and RL (Reflection Loss) performance
   - Provide prediction confidence assessment

### Validation Result Types

#### Experimental Data (🧪 Green Border)
- **Source**: Literature database
- **Confidence**: High
- **Information**: Specific performance parameters, DOI links, experimental synthesis methods, testing procedures

#### AI Prediction (🔮 Orange Border)
- **Source**: PLS prediction model
- **Confidence**: Medium-Low
- **Information**: EAB/RL classification predictions, confidence scores, performance explanations

#### Recommendation Only (📋 Gray Border)
- **Source**: RAG recommendation system
- **Confidence**: Basic
- **Information**: Recommendation source and relevance, literature descriptions

## Performance Metrics

### EAB (Effective Absorption Bandwidth)
- **0**: Poor - ≤ 4 GHz
- **1**: Fair - 4-8 GHz
- **2**: Good - 8-12 GHz
- **3**: Excellent - > 12 GHz

### RL (Reflection Loss)
- **0**: Excellent - ≤ -50 dB
- **1**: Good - -50 ~ -20 dB
- **2**: Fair - -20 ~ -10 dB
- **3**: Poor - > -10 dB

## Troubleshooting

### Common Issues

#### Program Closes Immediately After Startup
**Solutions:**
1. Use debug startup: `python debug_start.py`
2. Check logs: `logs/app.log`
3. Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

#### MaterialValidator Initialization Failed
**Solution:**
```bash
pip install scikit-learn pandas numpy
```

#### Sentence Embedding Model Download Failed
**Solutions:**
1. Check network connection
2. Use mirror source: `pip install sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple/`

#### ChromaDB Database Error
**Solution:**
```bash
# Delete vector database and reinitialize
Remove-Item -Recurse recommend/data/vectordb
```

### Complete Reset
If problems persist:
```bash
# 1. Clean cache and database
Remove-Item -Recurse recommend/data/vectordb -ErrorAction SilentlyContinue
Remove-Item -Recurse recommend/logs -ErrorAction SilentlyContinue

# 2. Reinstall dependencies
pip uninstall -y sentence-transformers chromadb
pip install sentence-transformers chromadb

# 3. Start with debug mode
python debug_start.py
```

## Technology Stack

### Offline Components
- **PyQt5**: User interface framework
- **ChromaDB**: Local vector database
- **SentenceTransformer**: Text embedding model
- **scikit-learn**: Machine learning prediction
- **pandas/numpy**: Data processing

### Optional Online Components
- **OpenAI API**: Intelligent chat functionality

## System Advantages

- **🔌 Offline First**: Core functions run completely locally
- **🚀 High Performance**: Fast local vector database retrieval
- **🧠 Intelligent Prediction**: Integrated machine learning prediction models
- **📊 Rich Data**: Based on real literature data with complete traceability 