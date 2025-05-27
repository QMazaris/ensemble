# AI Pipeline Dashboard

A comprehensive machine learning pipeline with an interactive Streamlit dashboard for model evaluation, comparison, and optimization. The pipeline supports both k-fold cross-validation and single split evaluation modes, with consistent cost calculations across all model types.

## Features

- **Interactive Dashboard**: Streamlit-based interface for configuring and monitoring the pipeline
- **Flexible Training Modes**:
  - K-Fold Cross Validation (default: 5 folds)
  - Single Split (80/20 train/test)
- **Advanced Model Optimization**:
  - Bayesian optimization using Optuna for hyperparameter tuning
  - Configurable number of optimization iterations (default: 50)
  - Parameter spaces defined in config.py for easy modification
- **Consistent Cost Calculation**:
  - K-Fold Mode: Costs calculated on full dataset and divided by N_SPLITS
  - Single Split Mode: Separate costs for train/test splits, with full cost as sum
  - Fair comparison across all model types
- **Feature Engineering**:
  - Variance filtering
  - Correlation filtering
  - Automatic feature encoding
- **Comprehensive Evaluation**:
  - Cost-optimized and accuracy-optimized thresholds
  - Performance metrics (precision, recall, accuracy)
  - Confusion matrices
  - Class balance visualization
- **Model Management**:
  - Support for multiple model types (XGBoost, RandomForest, etc.)
  - Base model integration (AD_Decision, CL_Decision, AD_or_CL_Fail)
  - Model persistence and loading
- **Visualization**:
  - Threshold sweep plots
  - Model comparison plots
  - Class distribution plots
  - Performance summary plots

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Dashboard

1. Activate your virtual environment
2. Run the Streamlit app:
```bash
streamlit run app.py
```

### Configuring Settings

The dashboard sidebar allows you to configure:

1. **Model Settings**:
   - Select models to evaluate
   - Enable/disable hyperparameter optimization
   - Set number of optimization iterations (10-500)
   - Choose evaluation mode (K-Fold or Single Split)
   - Set number of folds (when K-Fold is enabled)

2. **Feature Settings**:
   - Enable/disable variance filtering
   - Set variance threshold
   - Enable/disable correlation filtering
   - Set correlation threshold

3. **Cost Settings**:
   - Set false positive cost (C_FP)
   - Set false negative cost (C_FN)

### Pipeline Components

1. **Data Preparation**:
   - Load and preprocess data
   - Apply feature filters
   - Encode categorical variables

2. **Model Training**:
   - K-Fold Mode:
     * Split data into N folds
     * For each fold: optimize hyperparameters, train model, evaluate on test fold
     * Average results across folds
   - Single Split Mode:
     * Split data into train/test (80/20)
     * Optimize hyperparameters on training set
     * Train and evaluate on respective splits

3. **Cost Calculation**:
   - K-Fold Mode:
     * Base Models: Full dataset cost divided by N_SPLITS
     * Meta Models: Average cost across folds
     * AD/CL Models: Average cost across folds
   - Single Split Mode:
     * Base Models: Separate costs for train/test, full cost as sum
     * Meta Models: Costs on train/test splits, full cost as sum
     * AD/CL Models: Costs on train/test splits, full cost as sum

4. **Model Evaluation**:
   - Calculate performance metrics
   - Generate threshold sweep plots
   - Create model comparison plots
   - Export metrics for dashboard visualization

### Dashboard Tabs

1. **Overview**:
   - Class distribution visualization
   - Dataset statistics
   - Feature information

2. **Model Performance**:
   - Cost-optimized and accuracy-optimized results
   - Performance metrics by model and split
   - Threshold sweep visualizations

3. **Model Comparison**:
   - Side-by-side model comparisons
   - Cost and accuracy trade-offs
   - Performance across different splits

## Project Structure

```
├── app.py                 # Streamlit dashboard application
├── run.py                 # Main pipeline execution script
├── config.py             # Configuration settings
├── requirements.txt      # Project dependencies
├── helpers/             # Helper modules
│   ├── __init__.py
│   ├── data.py          # Data preparation functions
│   ├── modeling.py      # Model training and evaluation
│   └── plotting.py      # Visualization functions
├── tabs/                # Dashboard tab modules
│   ├── __init__.py
│   ├── overview.py      # Overview tab
│   └── performance.py   # Performance tab
└── output/             # Output directories
    ├── models/         # Saved models
    ├── plots/          # Generated plots
    └── predictions/    # Model predictions
```

## Adding New Models

To add a new model to the pipeline:

1. Add the model definition to `config.py` in the `MODELS` dictionary
2. Ensure the model follows scikit-learn's estimator interface
3. Define appropriate hyperparameter space in `config.py`
4. The pipeline will automatically include the new model in evaluation

## Best Practices

1. **Cost Settings**:
   - Set appropriate C_FP and C_FN values based on business requirements
   - Consider the impact of cost ratios on model selection

2. **Evaluation Mode**:
   - Use K-Fold for more robust performance estimates
   - Use Single Split for faster iteration and when data is limited

3. **Hyperparameter Optimization**:
   - Start with 50 iterations for quick results
   - Increase iterations for more thorough optimization
   - Monitor optimization progress in the dashboard

4. **Feature Selection**:
   - Use variance filtering to remove low-variance features
   - Apply correlation filtering to reduce feature redundancy
   - Monitor feature importance in model outputs

## Troubleshooting

1. **Memory Issues**:
   - Reduce number of optimization iterations
   - Decrease number of folds
   - Use feature filtering to reduce dimensionality

2. **Long Training Times**:
   - Disable hyperparameter optimization
   - Use single split mode
   - Reduce number of models in evaluation

3. **Model Performance**:
   - Check class balance
   - Adjust cost settings
   - Review feature selection thresholds

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