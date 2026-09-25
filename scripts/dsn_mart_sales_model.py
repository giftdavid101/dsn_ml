"""
DSN Mart Sales Prediction
===========================
Predicts `total_sales` for product-store combinations.

Pipeline:
1. Load train/test, combine for consistent cleaning
2. Clean inconsistent category text (case duplicates)
3. Impute missing values:
     - product_weight_kg -> median weight of the SAME product elsewhere
     - store_size         -> "Unknown" (missing for 3 entire stores, not recoverable)
4. Engineer 2 extra features: price_per_kg, store_profile (format+size combo)
5. Encode: numeric passthrough, ordinal for store_size/tier, one-hot for the rest
6. Train a RandomForestRegressor (tuned via 5-fold CV on RMSE)
7. Predict on test set, write submission.csv
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score

RANDOM_STATE = 42

# ---------- 1. Load ----------
train = pd.read_csv('/mnt/user-data/uploads/train.csv')
test = pd.read_csv('/mnt/user-data/uploads/test.csv')

train['is_train'] = 1
test['is_train'] = 0
test['total_sales'] = np.nan
full = pd.concat([train, test], sort=False, ignore_index=True)

# ---------- 2. Clean inconsistent text categories ----------
full['product_category'] = full['product_category'].str.strip().str.title()
full['fat_content'] = full['fat_content'].str.strip().str.title()

# ---------- 3. Impute missing values ----------
# product_weight_kg: same product should weigh roughly the same everywhere it appears
prod_weight = full.groupby('product_code')['product_weight_kg'].transform('median')
full['product_weight_kg'] = full['product_weight_kg'].fillna(prod_weight)
cat_weight = full.groupby('product_category')['product_weight_kg'].transform('median')
full['product_weight_kg'] = full['product_weight_kg'].fillna(cat_weight)  # fallback

# store_size: missing for 3 entire stores -> genuinely unknown, keep as its own category
full['store_size'] = full['store_size'].fillna('Unknown')

# flag exact-zero shelf visibility (likely "not recorded" rather than a true zero)
full['visibility_was_zero'] = (full['shelf_visibility'] == 0).astype(int)

# ---------- 4. Feature engineering ----------
full['price_per_kg'] = full['product_price'] / full['product_weight_kg']
full['store_profile'] = full['store_format'] + '_' + full['store_size']

# ---------- 5. Define feature groups & encoders ----------
numeric_cols = ['product_weight_kg', 'shelf_visibility', 'product_price',
                 'store_age_years', 'visibility_was_zero', 'price_per_kg']
ordinal_cols = ['store_size', 'store_location_tier']
nominal_cols = ['fat_content', 'product_category', 'store_code', 'store_format', 'store_profile']
features = numeric_cols + ordinal_cols + nominal_cols

size_order = ['Unknown', 'Small', 'Medium', 'Large']
tier_order = ['Tier_1', 'Tier_2', 'Tier_3']

preprocess = ColumnTransformer([
    ('num', 'passthrough', numeric_cols),
    ('ord', OrdinalEncoder(categories=[size_order, tier_order]), ordinal_cols),
    ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_cols),
])

model = RandomForestRegressor(
    n_estimators=600, max_depth=6, min_samples_leaf=8,
    max_features=0.6, random_state=RANDOM_STATE, n_jobs=-1
)

pipe = Pipeline([('prep', preprocess), ('model', model)])

# ---------- 6. Validate ----------
train_data = full[full['is_train'] == 1]
X, y = train_data[features], train_data['total_sales']

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_rmse = -cross_val_score(pipe, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
print(f"Cross-validated RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std():.2f})")

# ---------- 7. Fit on full train, predict test ----------
pipe.fit(X, y)

test_data = full[full['is_train'] == 0]
preds = pipe.predict(test_data[features])
preds = np.clip(preds, 0, None)  # sales can't be negative

submission = pd.DataFrame({
    'id': test_data['id'],
    'total_sales': preds
})
submission.to_csv('outputs/submission.csv', index=False)
print("\nSaved submission.csv:")
print(submission.head())

# ---------- Feature importance (for understanding what drives sales) ----------
ohe = pipe.named_steps['prep'].named_transformers_['nom']
ohe_names = ohe.get_feature_names_out(nominal_cols)
all_names = numeric_cols + ordinal_cols + list(ohe_names)
importances = pipe.named_steps['model'].feature_importances_
imp_df = pd.DataFrame({'feature': all_names, 'importance': importances}).sort_values('importance', ascending=False)
print("\nTop 10 most important features:")
print(imp_df.head(10).to_string(index=False))
