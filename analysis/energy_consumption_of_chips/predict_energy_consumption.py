# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Read the data
df = pd.read_csv('epoch_data.csv')

# Extract relevant columns
# Use FP16 Performance if available, otherwise use FP32 Performance
features_cols = ['Die Size in mm^2', 'Number of transistors in million', 'FP16 Performance (FLOP/s)']
target_col = 'TDP in W'

# Create a copy with relevant columns
data = df[features_cols + [target_col]].copy()

# Fill missing FP16 with FP32 values where available
if 'FP32 Performance (FLOP/s)' in df.columns:
    data['FP16 Performance (FLOP/s)'].fillna(df['FP32 Performance (FLOP/s)'], inplace=True)

# Drop rows with any missing values
data_clean = data.dropna()

print(f"Dataset summary:")
print(f"Total rows in original data: {len(df)}")
print(f"Rows with complete data: {len(data_clean)}")
print(f"\nFeature statistics:")
print(data_clean.describe())

# Prepare features and target
X = data_clean[features_cols].values
y = data_clean[target_col].values

# Log-transform FLOPs to handle large scale differences
X_transformed = X.copy()
X_transformed[:, 2] = np.log10(X_transformed[:, 2] + 1)  # Log10 of FLOP/s

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}

print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80)

best_model = None
best_score = -np.inf
best_name = ""

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"\n{name}:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  MAE: {mae:.2f} W")
    print(f"  RMSE: {rmse:.2f} W")

    # Track best model
    if test_r2 > best_score:
        best_score = test_r2
        best_model = model
        best_name = name

print(f"\nBest model: {best_name} (Test R² = {best_score:.4f})")

# Feature importance for best model (if available)
if hasattr(best_model, 'feature_importances_'):
    print(f"\nFeature Importances ({best_name}):")
    feature_names = ['Die Size (mm²)', 'Transistor Count (M)', 'Log10(FP16 FLOP/s)']
    importances = best_model.feature_importances_
    for name, importance in zip(feature_names, importances):
        print(f"  {name}: {importance:.4f}")

# Visualization: Predicted vs Actual
y_pred_best = best_model.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Predicted vs Actual
axes[0].scatter(y_test, y_pred_best, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual TDP (W)', fontsize=12)
axes[0].set_ylabel('Predicted TDP (W)', fontsize=12)
axes[0].set_title(f'Predicted vs Actual Energy Consumption\n{best_name} (R² = {best_score:.4f})', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3, linestyle='--')

# Plot 2: Residuals
residuals = y_test - y_pred_best
axes[1].scatter(y_pred_best, residuals, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted TDP (W)', fontsize=12)
axes[1].set_ylabel('Residuals (W)', fontsize=12)
axes[1].set_title(f'Residual Plot\n{best_name}', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('energy_prediction_model.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved as 'energy_prediction_model.png'")

# Example prediction function
def predict_tdp(die_size, transistor_count, fp16_flops, model=best_model):
    """
    Predict TDP given chip specifications.

    Parameters:
    - die_size: Die size in mm²
    - transistor_count: Number of transistors in millions
    - fp16_flops: FP16 performance in FLOP/s

    Returns:
    - Predicted TDP in Watts
    """
    X_new = np.array([[die_size, transistor_count, np.log10(fp16_flops + 1)]])
    return model.predict(X_new)[0]

# Example predictions
print("\n" + "="*80)
print("EXAMPLE PREDICTIONS")
print("="*80)

examples = [
    {"name": "Small chip", "die_size": 300, "transistors": 15000, "fp16_flops": 5e13},
    {"name": "Medium chip", "die_size": 500, "transistors": 30000, "fp16_flops": 1e14},
    {"name": "Large chip", "die_size": 800, "transistors": 80000, "fp16_flops": 3e14},
]

for ex in examples:
    pred_tdp = predict_tdp(ex["die_size"], ex["transistors"], ex["fp16_flops"])
    print(f"\n{ex['name']}:")
    print(f"  Die Size: {ex['die_size']} mm²")
    print(f"  Transistors: {ex['transistors']} million")
    print(f"  FP16 Performance: {ex['fp16_flops']:.2e} FLOP/s")
    print(f"  Predicted TDP: {pred_tdp:.1f} W")

# Don't show plot interactively
# plt.show()
