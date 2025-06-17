import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# Step 1: Generate dummy classification data
X, y = make_classification(
    n_samples=100,
    n_features=10,
    n_informative=5,
    n_classes=2,
    random_state=42
)

# Step 2: Train a simple XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

# Step 3: Convert to ONNX using onnxmltools
initial_type = [('input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_xgboost(model, initial_types=initial_type)

# Step 4: Save the ONNX model
with open("simple_xgb.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✓ ONNX model saved as 'simple_xgb.onnx'")
