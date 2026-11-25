# Model Comparison Results

_Generated automatically by src/train.py - all numbers from real model runs._


| Model | Train R2 | Test R2 | CV R2 (mean+-std) | Test MAE | Test RMSE | Overfit? |
|---|---|---|---|---|---|---|
| Linear Regression | 0.9324 | 0.9333 | 0.9313+-0.0033 | 14.86 | 21.94 | No |
| Random Forest | 0.9895 | 0.9267 | 0.9182+-0.0034 | 17.45 | 23.00 | YES |
| XGBoost | 0.9990 | 0.9721 | 0.9686+-0.0013 | 10.53 | 14.20 | No |