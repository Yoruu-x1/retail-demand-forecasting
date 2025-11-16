import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from datetime import timedelta

DATA_PATH = "Retail_Dataset2.csv"
MODEL_PATH = "model.pkl"
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    """Load dataset mentah dari file CSV."""
    df = pd.read_csv(path)
    return df

# CLEANING DATA & AGGREGATION
def clean_and_aggregate(df):
    """
    Membersihkan data, konversi tanggal, membersihkan Order_Demand,
    lalu agregasi menjadi total sales per hari per produk per gudang.
    """

    # Convert kolom Date ke datetime 
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

    # Membersihkan nilai Order_Demand
    df['Order_Demand'] = (
        df['Order_Demand'].astype(str)
        .str.replace(',', '')   
        .str.replace(' ', '')   
        .replace('nan', np.nan) 
    )
    df['Order_Demand'] = pd.to_numeric(df['Order_Demand'], errors='coerce').fillna(0)

    # Agregasi ke level time series: satu baris per hari per product/warehouse/category
    agg = df.groupby(
        ['Date', 'Product_Code', 'Warehouse', 'Product_Category'],
        as_index=False
    )['Order_Demand'].sum().rename(columns={'Order_Demand': 'sales'})

    return agg

# FEATURE ENGINEERING
def create_features(df):
    """
    Membuat fitur-fitur time series:
    - fitur kalender
    - rata-rata sales per produk
    - label encoding untuk kategori
    - lag features (1,7,14,28 hari)
    - rolling features (mean, std, min, max)
    - trend index
    """

    df = df.sort_values('Date').copy()

    # ======== Fitur tanggal ========
    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # ======== Statistik rata-rata per produk ========
    prod_mean = df.groupby('Product_Code')['sales'].mean().rename('prod_sales_mean')
    df = df.merge(prod_mean, on='Product_Code', how='left')

    # ======== Label Encoding untuk kategori ========
    le_wh = LabelEncoder()
    df['Warehouse_le'] = le_wh.fit_transform(df['Warehouse'])

    le_cat = LabelEncoder()
    df['Product_Category_le'] = le_cat.fit_transform(df['Product_Category'])

    # ======== Buat lag & rolling per kombinasi (produk, gudang) ========
    df_list = []
    for (prod, wh), g in df.groupby(['Product_Code', 'Warehouse']):
        g = g.sort_values('Date').set_index('Date')

        # ----- Fitur Lag -----
        g['lag_1']  = g['sales'].shift(1)
        g['lag_7']  = g['sales'].shift(7)
        g['lag_14'] = g['sales'].shift(14)
        g['lag_28'] = g['sales'].shift(28)

        # ----- Rolling Mean -----
        g['rmean_7']  = g['sales'].shift(1).rolling(7).mean()
        g['rmean_28'] = g['sales'].shift(1).rolling(28).mean()

        # ----- Rolling Std, Min, Max -----
        g['rstd_7']  = g['sales'].shift(1).rolling(7).std().fillna(0)
        g['rstd_28'] = g['sales'].shift(1).rolling(28).std().fillna(0)
        g['rmin_7']  = g['sales'].shift(1).rolling(7).min().fillna(0)
        g['rmax_7']  = g['sales'].shift(1).rolling(7).max().fillna(0)

        # ----- Trend Index -----
        g['trend'] = np.arange(len(g))

        df_list.append(g.reset_index())

    res = pd.concat(df_list, axis=0).reset_index(drop=True)

    # Hapus baris yang tidak memiliki lag (karena NA)
    res = res.dropna(subset=['lag_1', 'lag_7', 'lag_14', 'lag_28', 'rmean_7', 'rmean_28'])

    return res, le_wh, le_cat


# ============================================================
# MODEL TRAINING
# ============================================================
def train_model(df, le_wh, le_cat):
    """Melatih LightGBM menggunakan fitur time series."""

    # Daftar fitur final untuk training
    features = [
        'day', 'month', 'year', 'dayofweek', 'is_weekend',
        'prod_sales_mean', 'Warehouse_le', 'Product_Category_le',
        'lag_1', 'lag_7', 'lag_14', 'lag_28',
        'rmean_7', 'rmean_28',
        'rstd_7', 'rstd_28', 'rmin_7', 'rmax_7',
        'trend'
    ]

    target = 'sales'
    df = df.sort_values('Date')

    # ======== Time-based Train/Test Split (30 hari terakhir sebagai test) ========
    last_date = df['Date'].max()
    cutoff = last_date - timedelta(days=30)

    train_df = df[df['Date'] <= cutoff].copy()
    test_df  = df[df['Date'] > cutoff].copy()

    X_train = train_df[features]
    y_train = train_df[target]
    X_test  = test_df[features]
    y_test  = test_df[target]

    # ======== LightGBM Dataset ========
    lgb_train = lgb.Dataset(X_train, label=y_train.values)
    lgb_eval  = lgb.Dataset(X_test,  label=y_test.values)

    # ======== Parameter Model ========
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'seed': 42,
        'verbosity': -1
    }

    # ======== Train LightGBM (with early stopping) ========
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_eval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )

    # ======== Evaluasi ========
    preds = model.predict(X_test, num_iteration=model.best_iteration)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = (np.abs((y_test - preds) / y_test.replace(0,1))).mean() * 100

    print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")

    # ======== Simpan model & encoder ========
    save_path = os.path.join(ARTIFACTS_DIR, MODEL_PATH)
    joblib.dump(model, save_path)
    joblib.dump(le_wh,  os.path.join(ARTIFACTS_DIR, 'le_wh.pkl'))
    joblib.dump(le_cat, os.path.join(ARTIFACTS_DIR, 'le_cat.pkl'))

    # Simpan dataset hasil feature engineering
    df.to_csv(os.path.join(ARTIFACTS_DIR, 'processed.csv'), index=False)

    print("MODEL SAVED:", save_path)

    return model

def main():
    print("\n=== LOADING DATA ===")
    df = load_data()

    print("=== CLEAN & AGGREGATE ===")
    agg = clean_and_aggregate(df)

    print("=== FEATURE ENGINEERING ===")
    df_feat, le_wh, le_cat = create_features(agg)

    print("=== TRAINING MODEL ===")
    train_model(df_feat, le_wh, le_cat)


if __name__ == "__main__":
    main()
