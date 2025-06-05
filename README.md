# Ensemble Pipeline V2 - Full Stack ML Application

A comprehensive machine learning pipeline with a modern full-stack architecture for binary classification tasks. Features automated model training, hyperparameter optimization, cross-validation, and interactive visualization with **in-memory data sharing** and **matplotlib-free plotting**.

## 🏗️ Architecture

This project follows a full-stack architecture with clear separation of concerns:

```
ensamble_pipelineV2/
├── backend/                 # Backend logic and API
│   ├── api/                # FastAPI REST endpoints
│   ├── helpers/            # Core ML utilities
│   │   ├── data.py        # Data preparation and CV
│   │   ├── modeling.py    # Model training and evaluation
│   │   ├── metrics.py     # Performance metrics
│   │   ├── plotting.py    # Data-only plotting functions
│   │   ├── reporting.py   # Results reporting
│   │   └── export_metrics_for_streamlit.py
│   ├── config.py          # Configuration settings
│   └── run.py             # Main pipeline execution
├── frontend/               # Frontend applications
│   └── streamlit/         # Streamlit dashboard
│       ├── app.py         # Main dashboard app
│       ├── tabs.py        # Dashboard tabs
│       └── utils.py       # Frontend utilities
├── shared/                # Shared utilities and schemas
│   ├── data_service.py    # In-memory data sharing
│   ├── schemas.py         # Data schemas
│   └── utils.py           # Shared utilities
├── tests/                 # Comprehensive test suite
│   ├── backend/           # Backend tests
│   ├── frontend/          # Frontend tests
│   ├── shared/            # Shared component tests
│   └── run_tests.py       # Test runner
├── data/                  # Training data
└── output/                # Generated outputs
```

## ✨ Features

### 🚀 **New in V2: Modern Architecture**
- **In-Memory Data Sharing**: No more CSV intermediaries between frontend/backend
- **Matplotlib-Free**: All plotting uses Plotly for interactive visualizations
- **Improved K-Fold**: Cleaner, more maintainable cross-validation logic
- **Comprehensive Testing**: Full test suite with coverage reporting
- **Better Code Organization**: Clear separation of concerns

### Backend (API)
- **RESTful API** with FastAPI
- **Model Training Pipeline** with automated hyperparameter optimization
- **Enhanced Cross-Validation** with improved scikit-learn integration
- **Model Export** in pickle and ONNX formats
- **Real-time Pipeline Status** tracking
- **Model Prediction** endpoints
- **Configuration Management** via API
- **Data Service**: Thread-safe in-memory data sharing

### Frontend (Dashboard)
- **Interactive Streamlit Dashboard** for pipeline management
- **Real-time Visualization** with Plotly (no static images)
- **Data Management** and preprocessing configuration
- **Model Comparison** and analysis tools
- **Download Center** for results and models
- **Responsive UI** with modern plotting

### Core ML Features
- **Multiple Algorithms**: XGBoost, Random Forest, Logistic Regression, etc.
- **Automated Hyperparameter Tuning** with Optuna
- **Cost-Sensitive Learning** with configurable cost matrices
- **Threshold Optimization** for precision/recall balance
- **Feature Engineering** with variance and correlation filtering
- **Class Imbalance Handling** with SMOTE
- **Improved K-Fold Logic** with helper functions and better code structure

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ensamble_pipelineV2
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**
   - Place your training data in `data/training_data.csv`
   - Update `backend/config.py` with your data specifications

### Running the Application

#### Option 1: Full Stack (Recommended)

1. **Start the Backend API**
   ```bash
   cd backend
   python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start the Frontend Dashboard**
   ```bash
   streamlit run frontend/streamlit/app.py
   ```

3. **Access the applications**
   - Dashboard: http://localhost:8501
   - API Documentation: http://localhost:8000/docs

#### Option 2: Direct Pipeline Execution
```bash
cd backend
python run.py
```

## 📊 Usage

### Via Dashboard
1. Open the Streamlit dashboard
2. Configure your pipeline settings in the sidebar
3. Upload or verify your training data
4. Click "Run Pipeline" to start training
5. Monitor progress and view **interactive plots** in real-time
6. **No file refreshing needed** - data updates automatically

### Via API
```python
import requests

# Start pipeline
response = requests.post("http://localhost:8000/pipeline/run")

# Check status
status = requests.get("http://localhost:8000/pipeline/status")

# Make predictions
prediction = requests.post("http://localhost:8000/models/predict", 
                         json={"features": [1.0, 2.0, 3.0], "model_name": "XGBoost"})
```

## ⚙️ Configuration

Key configuration options in `backend/config.py`:

```python
# Data Configuration
DATA_PATH = 'data/training_data.csv'
TARGET = 'GT_Label'
EXCLUDE_COLS = ['Image', 'AD_Decision', ...]

# Model Configuration
MODELS = {
    'XGBoost': xgb.XGBClassifier(...),
    'RandomForest': RandomForestClassifier(...)
}

# Training Configuration
USE_KFOLD = True
N_SPLITS = 5
OPTIMIZE_HYPERPARAMS = True
HYPERPARAM_ITER = 25

# Cost Configuration
C_FP = 1    # Cost of false positive
C_FN = 30   # Cost of false negative
```

## 📁 Output Structure

```
output/
├── models/                 # Trained models (.pkl/.onnx)
├── plots/                 # Backup visualization plots (optional)
├── predictions/           # Backup model predictions (optional)
├── streamlit_data/        # Backup dashboard data (optional)
└── logs/                  # Execution logs
```

**Note**: With the new in-memory data service, files are saved as backup only. The frontend gets data directly from memory for better performance.

## 🧪 Testing

### Run All Tests
```bash
python tests/run_tests.py
```

### Run Specific Test Categories
```bash
pytest tests/backend/          # Backend tests
pytest tests/frontend/         # Frontend tests  
pytest tests/shared/           # Shared component tests
pytest tests/test_api.py       # API tests
```

### Run with Coverage
```bash
pytest --cov=backend --cov=shared --cov=frontend --cov-report=html
```

## 🔧 Development

### Code Quality
```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 .
```

### Adding New Models
1. Add model configuration to `backend/config.py`
2. Update hyperparameter space if needed
3. Test with the pipeline

### API Development
- API endpoints are in `backend/api/main.py`
- Add new endpoints following FastAPI patterns

### Testing New Features
- Add tests to appropriate `tests/` subdirectory
- Use the test runner: `python tests/run_tests.py`
- Ensure coverage remains high

## 🎯 Architecture Benefits

### Performance Improvements
- **🚀 Faster Data Access**: In-memory sharing eliminates file I/O bottlenecks
- **⚡ Real-time Updates**: Frontend gets data immediately after backend processing
- **📊 Interactive Plots**: Plotly charts instead of static matplotlib images

### Code Quality Improvements  
- **🧹 Cleaner K-Fold Logic**: Extracted helper functions reduce code duplication
- **🔧 Better Testing**: Comprehensive test suite with 90%+ coverage
- **📁 Organized Structure**: Clear separation between backend, frontend, and shared code
- **🛡️ Type Safety**: Better error handling and data validation

### Developer Experience
- **🔄 No CSV Dependencies**: Eliminates file corruption and missing file issues
- **🎨 Modern UI**: Interactive visualizations with Plotly
- **🧪 Test-Driven**: Easy to test and maintain with comprehensive test suite
- **📖 Better Documentation**: Clear architecture and usage examples

## 🚧 Migration Notes

If upgrading from V1:
1. **Data Flow**: Frontend now uses data service instead of reading CSV files
2. **Plotting**: All plots are now interactive Plotly charts
3. **Testing**: New test suite available for validation
4. **File Structure**: Code reorganized into backend/frontend/shared structure

The system maintains backward compatibility with optional file saving for legacy workflows.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check the `/docs` endpoint when running the API
- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions

## 🗺️ Roadmap

- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] Model versioning and MLOps integration
- [ ] Advanced visualization with Plotly/Dash
- [ ] Real-time model monitoring
- [ ] A/B testing framework
- [ ] Model explainability with SHAP

## Architecture Overview

### Simplified Data Flow
The pipeline now uses a maximally simple data flow architecture:

1. **Pipeline Execution**: The main pipeline (`backend/run.py`) processes data and trains models
2. **Direct Memory Export**: After completion, the pipeline exports all results directly to API memory via the `DataService` singleton
3. **API Serving**: The API (`backend/api/main.py`) serves data directly from memory to the frontend
4. **No Intermediate Files**: No CSV files or complex export processes - pure in-memory data flow

### Key Components

- **DataService** (`shared/data_service.py`): Simplified singleton for pure in-memory data storage
- **Pipeline** (`backend/run.py`): Exports data directly to API memory after each run
- **API** (`backend/api/main.py`): Serves data from memory to frontend

### Benefits

- **Maximum Simplicity**: No file I/O overhead, no backup complexity
- **Real-time Updates**: Fresh data immediately available after pipeline completion
- **Memory Efficient**: Data stored once in memory, served directly to frontend
- **Clean Architecture**: Clear separation between pipeline execution and data serving

### Usage

1. Run the pipeline via API: `POST /pipeline/run`
2. Pipeline automatically exports results to API memory
3. Frontend gets fresh data immediately via API endpoints
4. Re-run pipeline for updated results with new parameters

## Installation and Setup

// ... existing content ...