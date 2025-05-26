# AI Pipeline Dashboard

A comprehensive machine learning pipeline with a Streamlit dashboard for model training, evaluation, and analysis. This project implements a flexible pipeline that supports both single-split and k-fold cross-validation approaches, with Bayesian hyperparameter optimization using Optuna.

## Features

- **Interactive Dashboard**: Streamlit-based UI for easy configuration and monitoring
- **Flexible Training Modes**:
  - Single train-test split
  - K-fold cross-validation (2-10 folds)
- **Advanced Model Optimization**:
  - Bayesian hyperparameter optimization using Optuna
  - Configurable number of optimization iterations
  - Support for multiple model types (XGBoost, RandomForest, etc.)
- **Feature Engineering**:
  - Optional variance-based feature filtering
  - Optional correlation-based feature filtering
- **Comprehensive Evaluation**:
  - Cost-based optimization
  - Accuracy-based optimization
  - Detailed performance metrics
  - Visualization of results
- **Model Management**:
  - Save/load trained models
  - Export predictions and metrics
  - Production model optimization

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd ensamble_pipelineV2
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Dashboard

Run the Streamlit app using the virtual environment's Python:
```bash
.\venv\Scripts\streamlit run app.py --server.address 0.0.0.0
```

The dashboard will be available at `http://localhost:8501`

### Dashboard Configuration

The sidebar provides several configuration options:

#### Model Settings
- **Cost of False-Positive/Negative**: Set the cost weights for model evaluation
- **Use K-Fold Cross Validation**: Toggle between single split and k-fold modes
- **Number of K-Fold Splits**: Set the number of folds (2-10) when using k-fold

#### Feature Settings
- **Apply Feature Filtering**: Enable/disable feature selection
- **Variance Threshold**: Remove low-variance features
- **Correlation Threshold**: Remove highly correlated features

#### Optimization Settings
- **Optimize Hyperparameters**: Enable Bayesian optimization
- **Number of Optimization Iterations**: Set iterations (10-500) for hyperparameter search
- **Optimize Final Model**: Enable optimization for the production model

### Pipeline Components

1. **Data Preparation** (`helpers/data.py`):
   - Data loading and preprocessing
   - Feature engineering
   - Train-test splitting

2. **Model Training** (`helpers/modeling.py`):
   - Model initialization
   - Hyperparameter optimization
   - Training and evaluation

3. **Metrics and Evaluation** (`helpers/metrics.py`):
   - Performance metrics calculation
   - Threshold optimization
   - Cost-based evaluation

4. **Visualization** (`helpers/plotting.py`):
   - Threshold sweep plots
   - Model comparison plots
   - Performance visualizations

### Dashboard Tabs

1. **Overview**: Summary of model performance and key metrics
2. **Model Analysis**: Detailed analysis of model performance
3. **Plots Gallery**: Visualizations of model results
4. **Downloads**: Export models, predictions, and metrics

## Project Structure

```
ensamble_pipelineV2/
├── app.py                 # Streamlit dashboard
├── run.py                 # Main pipeline script
├── config.py             # Configuration settings
├── requirements.txt      # Project dependencies
├── helpers/
│   ├── data.py          # Data preparation
│   ├── modeling.py      # Model training and optimization
│   ├── metrics.py       # Performance metrics
│   ├── plotting.py      # Visualization
│   └── utils.py         # Utility functions
├── output/
│   ├── models/          # Saved models
│   ├── plots/           # Generated plots
│   └── predictions/     # Model predictions
└── tabs/                # Dashboard tab components
```

## Adding New Models

To add a new model to the pipeline:

1. Add the model to `config.py`:
```python
MODELS = {
    'YourModel': YourModelClass(),
    # ... existing models ...
}
```

2. Define hyperparameter space (optional):
```python
HYPERPARAM_SPACE = {
    'YourModel': {
        'param1': (min_value, max_value),
        'param2': [option1, option2, ...],
        # ... other parameters ...
    }
}
```

## Best Practices

1. **Virtual Environment**: Always use the virtual environment to ensure consistent dependencies
2. **Configuration**: Use the dashboard to modify settings rather than editing config.py directly
3. **Model Selection**: Start with simpler models before moving to complex ones
4. **Optimization**: 
   - Start with fewer iterations (50-100) for quick testing
   - Increase iterations (200-500) for final optimization
5. **Feature Selection**: Use feature filtering to reduce dimensionality and improve model performance

## Troubleshooting

1. **Import Errors**: Ensure you're using the virtual environment's Python
2. **Memory Issues**: Reduce batch size or number of features if running into memory constraints
3. **Optimization Time**: Adjust the number of iterations based on your time constraints
4. **Model Performance**: 
   - Check feature importance
   - Adjust cost weights
   - Try different hyperparameter spaces

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License Here]

## Contact

[Your Contact Information] 