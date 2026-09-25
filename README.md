# DSN Mart Sales Prediction

A machine learning solution for the **DSN Bootcamp Qualification Hackathon 2026 (ML Track)** — predicting `total_sales` for product-store combinations at DSN Mart, a Nigerian retail chain.

---

## Problem Statement

DSN Mart operates across Nigeria with stores of varying sizes, formats, and location tiers. The goal is to build a regression model that predicts the **total sales value** of a given product at a given store, based on historical retail data.

Accurate predictions help DSN Mart make better decisions around:

- Inventory allocation across stores
- Pricing and promotion strategy
- Store-level demand forecasting
- Product assortment planning

---

## Dataset

The dataset consists of historical product-store records split into train and test sets.

| Column | Description |
|---|---|
| `id` | Unique row identifier |
| `product_code` | Unique code for the product |
| `product_weight_kg` | Weight of the product in kilograms *(some missing)* |
| `fat_content` | Product label: `Low Fat` or `Regular` |
| `shelf_visibility` | Proportion of total display area allocated to this product |
| `product_category` | Product's category *(inconsistent capitalization)* |
| `product_price` | Product's listed price |
| `store_code` | Unique code for the store |
| `store_age_years` | Store operating age in years |
| `store_size` | Store size: `Small`, `Medium`, `Large` *(some missing)* |
| `store_location_tier` | `Tier_1` (major urban), `Tier_2` (state capitals), `Tier_3` (smaller towns) |
| `store_format` | `Corner Shop`, `Standard Supermarket`, `Superstore`, `Flagship Hypermarket` |
| `total_sales` | **Target** — total sales value for this product-store row *(train only)* |

**Files:**

- `train.csv` — historical records with `total_sales`
- `test.csv` — same schema, `total_sales` removed (to be predicted)
- `sample_submission.csv` — required submission format

---

## Evaluation Metric

**Root Mean Squared Error (RMSE)** — lower is better.

```
RMSE = sqrt( (1/n) * sum((y_true - y_pred)^2) )
```

---

## Solution Overview

1. **Data Cleaning** — standardize inconsistent `product_category` capitalization, normalize `fat_content`, impute missing `product_weight_kg` (median per category), impute missing `store_size` (as `"Unknown"`).
2. **Exploratory Data Analysis (EDA)** — target distribution, sales by store format/location tier/category, missing value analysis.
3. **Feature Engineering** — interaction features (`price_per_kg`, `shelf_x_price`, `shelf_x_weight`), group aggregations (`prod_sales_mean`, `store_sales_mean`, `pair_sales_mean`, `cat_sales_mean`), binned features.
4. **Encoding** — label encoding for low-cardinality categoricals, target encoding for `product_code` and `store_code`.
5. **Modeling** — LightGBM as primary model with early stopping; optional ensemble of XGBoost + CatBoost.
6. **Validation** — 5-fold K-Fold cross-validation with out-of-fold (OOF) predictions and a fixed random seed for reproducibility.

---

## Project Structure

```
dsn-mart-sales-prediction/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── src/
│   ├── clean.py            # Data cleaning utilities
│   ├── features.py         # Feature engineering functions
│   ├── train.py            # Model training pipeline
│   └── predict.py          # Generate submission
│
├── submissions/
│   └── submission.csv
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/dsn-mart-sales-prediction.git
cd dsn-mart-sales-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
python src/train.py --data-dir data/ --output submissions/submission.csv
```

Or explore the notebooks in order:

1. `01_eda.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_modeling.ipynb`

---

## Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
lightgbm>=4.0
xgboost>=2.0
catboost>=1.2
category_encoders>=2.6
matplotlib>=3.7
seaborn>=0.12
jupyter
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Model Performance

| Model | OOF RMSE | Notes |
|---|---|---|
| Baseline Linear Regression | TBD | Simple reference |
| Random Forest | TBD | Strong non-linear baseline |
| **LightGBM (final)** | TBD | Best single model |
| Ensemble (LGBM + XGB + CatBoost) | TBD | Small additional boost |

*Update this table with your actual cross-validation scores.*

---

## Key Insights

- **Product-store pair statistics** (`pair_sales_mean`) are the strongest predictors — the same product sells very differently across stores.
- **Store format and location tier** carry substantial signal about sales volume.
- **`product_category` required cleaning** — inconsistent capitalization created duplicate categories before standardization.
- **Missing values were informative** — the missingness pattern in `store_size` and `product_weight_kg` correlated with sales patterns.

---

## Competition Info

- **Competition**: [DSN Bootcamp Qualification Hackathon 2026 — ML Track](https://www.kaggle.com/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track)
- **Platform**: Kaggle
- **Task**: Regression — predict `total_sales`
- **Metric**: RMSE

---

## Acknowledgements

- **Data Science Nigeria (DSN)** for organizing the bootcamp and competition
- The Kaggle community for shared kernels and discussion insights
- Open-source libraries: LightGBM, XGBoost, CatBoost, scikit-learn

---

## License

This project is released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Author
Gift David
---

*Built as part of the DSN AI Bootcamp 2026 qualification process.*