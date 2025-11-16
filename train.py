# train.py
import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

DATA_PATH = "Retail_Dataset2.csv"  # pastikan nama file sesuai

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Basic processing...")
# date
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

# create date-derived features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek

# target
TARGET = "Order_Demand"

# define columns we will use (fixed order important)
FEATURE_ORDER = [
    "Open", "Promo", "SchoolHoliday", "Petrol_price",
    "Year", "Month", "Day", "DayOfWeek",
    "Product_Code", "Warehouse", "Product_Category", "StateHoliday"
]

# keep only existing features (safe)
FEATURE_ORDER = [f for f in FEATURE_ORDER if f in df.columns]

# ------------ handle missing values ------------
# categorical fillna -> 'MISSING'
cat_cols = ["Product_Code", "Warehouse", "Product_Category", "StateHoliday"]
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype(str).fillna("MISSING")

# numeric fillna -> 0 (or median could be used)
num_cols = [c for c in FEATURE_ORDER if c not in cat_cols]
df[num_cols] = df[num_cols].fillna(0)

# sanity: drop rows where target is missing or not numeric
df = df[pd.to_numeric(df[TARGET], errors="coerce").notnull()].copy()
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0)

# ------------ encode categoricals as integer mapping (and save mapping) ------------
encoders = {}
for c in cat_cols:
    if c in FEATURE_ORDER:
        vals = df[c].astype(str).tolist()
        # get unique preserving order of appearance for stable mapping
        uniq = list(dict.fromkeys(vals))
        mapping = {v: i for i, v in enumerate(uniq)}
        # map and store mapping + fallback mode
        df[c] = df[c].astype(str).map(mapping).fillna(0).astype(int)
        # mode (most frequent original string) for fallback
        mode_val = df[c].mode()
        # but we need original mode string:
        # compute original mode by value counts on original vals
        original_mode = pd.Series(vals).mode().iloc[0] if len(vals) > 0 else uniq[0]
        encoders[c] = {"mapping": mapping, "mode": original_mode}

# save encoders as JSON (mapping keys are strings)
with open(os.path.join(ARTIFACTS_DIR, "encoders.json"), "w", encoding="utf-8") as fh:
    json.dump(encoders, fh, ensure_ascii=False)

# save feature order
with open(os.path.join(ARTIFACTS_DIR, "features_order.json"), "w") as fh:
    json.dump(FEATURE_ORDER, fh)

# ------------ train/test split & model training ------------
X = df[FEATURE_ORDER].copy()
y = df[TARGET].copy()

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("Training XGBoost...")
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    use_label_encoder=False,
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# evaluate
preds = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
mae = float(mean_absolute_error(y_test, preds))
r2 = float(r2_score(y_test, preds))

print(f"Evaluation on test set -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")

# save model (joblib)
joblib.dump(model, os.path.join(ARTIFACTS_DIR, "model.pkl"))
# also copy model to root for compatibility
joblib.dump(model, "model.pkl")

# save simple eval summary
with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as fh:
    json.dump({"rmse": rmse, "mae": mae, "r2": r2}, fh)

print("Saved artifacts to", ARTIFACTS_DIR)
print("TRAINING COMPLETE.")
