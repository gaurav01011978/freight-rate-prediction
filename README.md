# Freight Rate Prediction – Spotter ML Assessment

Predicts truckload `posted_rate` using LightGBM.

## Results (time-based holdout: train Jan–Aug, test Sep–Oct 2025)
| Model | MAE ($) | MAPE |
|---|---|---|
| Baseline (median rate/mile × distance) | 175.08 | – |
| LightGBM | 30.04 | 1.2% |

## Project structure
```
freight_rate_model.ipynb     # full pipeline: EDA, cleaning, model, predictions
validation_predictions.csv   # final predictions (12,000 loads)
score.py                     # provided scorer
requirements.txt
data/                        # place the provided CSV files here
```

## How to run
1. Put `train_test.csv`, `validation.csv`, `validation_predictions_template.csv`, `december_chart_inputs.csv` inside `data/`.
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Open `freight_rate_model.ipynb`, set `DATA = "data/"` in cell 2 (it currently points to Google Drive), and run all cells. You can skip cell 1 (the Google Drive mount) outside Colab.
4. Validate outputs and create the December chart:
```bash
   python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_chart_inputs.csv
```

## Key decisions
- **Time-based split** because validation (Nov–Dec) is in the future.
- **Coordinates instead of city names** because 8 validation cities never appear in training.
- **Dropped `market_index`**: its relationship with rates shifts over time (MAE worsened from 30 to 76).
- **Cleaning**: negative weights → absolute value; ~1.4% extreme rate-per-mile outliers removed from training.
