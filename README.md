# Ensemble Pipeline V2 - Full Stack ML Application

A comprehensive machine learning pipeline with a modern full-stack architecture for binary classification tasks. Features automated model training, hyperparameter optimization, cross-validation, and interactive visualization.

## 🏗️ Architecture

This project follows a full-stack architecture with clear separation of concerns:

```
ensamble_pipelineV2/
├── backend/                 # Backend logic and API
│   ├── api/                # FastAPI REST endpoints
│   ├── helpers/            # Core ML utilities
│   ├── config.py          # Configuration settings
│   └── run.py             # Main pipeline execution
├── frontend/               # Frontend applications
│   └── streamlit/         # Streamlit dashboard
├── shared/                # Shared utilities and schemas
├── tests/                 # Test suite
├── data/                  # Training data
└── output/                # Generated outputs
```

## ✨ Features

### Backend (API)
- **RESTful API** with FastAPI
- **Model Training Pipeline** with automated hyperparameter optimization
- **Cross-Validation** with stratified k-fold
- **Model Export** in pickle and ONNX formats
- **Real-time Pipeline Status** tracking
- **Model Prediction** endpoints
- **Configuration Management** via API

### Frontend (Dashboard)
- **Interactive Streamlit Dashboard** for pipeline management
- **Real-time Visualization** of model performance
- **Data Management** and preprocessing configuration
- **Model Comparison** and analysis tools
- **Download Center** for results and models

### Core ML Features
- **Multiple Algorithms**: XGBoost, Random Forest, Logistic Regression, etc.
- **Automated Hyperparameter Tuning** with Optuna
- **Cost-Sensitive Learning** with configurable cost matrices
- **Threshold Optimization** for precision/recall balance
- **Feature Engineering** with variance and correlation filtering
- **Class Imbalance Handling** with SMOTE

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
5. Monitor progress and view results in real-time

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
├── plots/                 # Visualization plots
├── predictions/           # Model predictions
├── streamlit_data/        # Dashboard data
└── logs/                  # Execution logs
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run specific test categories:
```bash
pytest tests/test_api.py      # API tests
pytest tests/test_models.py   # Model tests
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
- Update tests in `tests/test_api.py`

## 📈 Performance Monitoring

The pipeline provides comprehensive metrics:
- **Model Performance**: Accuracy, Precision, Recall, F1-Score
- **Cost Analysis**: False Positive/Negative costs
- **Threshold Optimization**: ROC curves and cost curves
- **Cross-Validation**: Robust performance estimates
- **Feature Importance**: Model interpretability

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