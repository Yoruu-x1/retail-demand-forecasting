import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = "model.pkl"

@st.cache_data
def load_artifacts():
    """
    Memuat model, label encoder, dan processed dataset.
    Streamlit cache memastikan file tidak dimuat berulang ulang.
    """
    bundle = joblib.load(f"{ARTIFACTS_DIR}/{MODEL_PATH}")

    if isinstance(bundle, dict):
        model = bundle["model"]
        le_wh = bundle["le_wh"]
        le_cat = bundle["le_cat"]
    else:
       
        model = bundle
        le_wh = joblib.load(f"{ARTIFACTS_DIR}/le_wh.pkl")
        le_cat = joblib.load(f"{ARTIFACTS_DIR}/le_cat.pkl")

    processed = pd.read_csv(
        f"{ARTIFACTS_DIR}/processed.csv",
        parse_dates=['Date']
    )

    return model, le_wh, le_cat, processed

# FEATURE BUILDER
def build_feature_row(df_hist, product_code, warehouse, product_category, pred_date):
    """
    Membangun 1 baris fitur untuk tanggal prediksi tertentu.
    Menggunakan data historis + lag + rolling windows + trend.
    """

    # Ambil data historis per Product + Warehouse
    ser = df_hist[
        (df_hist['Product_Code'] == product_code) &
        (df_hist['Warehouse'] == warehouse)
    ].sort_values('Date')

    # Mapping tanggal → sales untuk akses cepat lag
    sales_map = ser.set_index('Date')['sales'].to_dict()

    # Helper untuk mengambil data lag, return 0 jika missing
    def get_lag(days):
        d = pred_date - timedelta(days=days)
        return sales_map.get(d, 0)

    # Window rolling (7 & 28 hari)
    last_7 = [sales_map.get(pred_date - timedelta(days=d), 0) for d in range(1, 8)]
    last_28 = [sales_map.get(pred_date - timedelta(days=d), 0) for d in range(1, 29)]

    # Trend = jumlah hari historis → semakin panjang histori semakin stabil
    trend_value = len(ser)

    # 19 fitur sesuai training
    row = {
        # --- Date features ---
        'day': pred_date.day,
        'month': pred_date.month,
        'year': pred_date.year,
        'dayofweek': pred_date.dayofweek,
        'is_weekend': int(pred_date.dayofweek in [5, 6]),

        # --- Statistical info per product ---
        'prod_sales_mean': ser['prod_sales_mean'].iloc[-1] if len(ser) > 0 else 0,

        # --- Encoded categorical ---
        'Warehouse_le': ser['Warehouse_le'].iloc[-1] if len(ser) > 0 else 0,
        'Product_Category_le': ser['Product_Category_le'].iloc[-1] if len(ser) > 0 else 0,

        # --- Lags ---
        'lag_1': get_lag(1),
        'lag_7': get_lag(7),
        'lag_14': get_lag(14),
        'lag_28': get_lag(28),

        # --- Rolling means ---
        'rmean_7': np.mean(last_7),
        'rmean_28': np.mean(last_28),

        # --- Rolling statistical (std/min/max) ---
        'rstd_7': np.std(last_7),
        'rstd_28': np.std(last_28),
        'rmin_7': min(last_7),
        'rmax_7': max(last_7),

        # --- Trend ---
        'trend': trend_value,
    }

    return pd.DataFrame([row])

# ITERATIVE FORECASTING
def iterative_forecast(model, df_hist, product_code, warehouse, product_category, start_date, horizon):
    """
    Melakukan peramalan satu per satu.
    Prediksi hari ke-n dipakai sebagai input lag untuk prediksi hari berikutnya.
    Supaya cocok dengan real forecasting, bukan naive independent prediction.
    """

    features = [
        'day','month','year','dayofweek','is_weekend',
        'prod_sales_mean','Warehouse_le','Product_Category_le',
        'lag_1','lag_7','lag_14','lag_28',
        'rmean_7','rmean_28',
        'rstd_7','rstd_28','rmin_7','rmax_7',
        'trend'
    ]

    preds = []
    cur_hist = df_hist.copy()   

    for h in range(horizon):
        pred_date = start_date + timedelta(days=h + 1)

        X_row = build_feature_row(cur_hist, product_code, warehouse, product_category, pred_date)

        X_row = X_row[features]

        # Prediksi model
        yhat = float(model.predict(X_row)[0])
        yhat = max(0, yhat)  # untuk mencegah nilai negatif

        # Simpan hasil prediksi
        preds.append({'Date': pred_date, 'pred': yhat})

        # Tambahkan prediksi sebagai data historis untuk iterasi selanjutnya
        new_row = {
            'Date': pred_date,
            'Product_Code': product_code,
            'Warehouse': warehouse,
            'Product_Category': product_category,
            'sales': yhat
        }
        cur_hist = pd.concat([cur_hist, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(preds)

# STREAMLIT UI
st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")
st.title("Retail Demand Forecasting")

model, le_wh, le_cat, processed = load_artifacts()

products = sorted(processed['Product_Code'].unique().tolist())
warehouses = sorted(processed['Warehouse'].unique().tolist())

product_cat_map = processed.groupby('Product_Code')['Product_Category'].first().to_dict()

# Input panel
col1, col2, col3 = st.columns(3)
with col1:
    prod_sel = st.selectbox("Pilih Product Code", products)
with col2:
    wh_sel = st.selectbox("Pilih Warehouse", warehouses)
with col3:
    horizon = st.number_input("Days to forecast (1–30)", min_value=1, max_value=30, value=7)

# Tanggal historical terakhir
last_date = processed['Date'].max()
st.write(f"Last historical date: **{last_date.date()}**")

# FORECAST BUTTON
if st.button("Forecast"):
    category = product_cat_map.get(prod_sel, processed['Product_Category'].iloc[0])

    # Jalankan iterative forecasting
    preds_df = iterative_forecast(model, processed, prod_sel, wh_sel, category, last_date, horizon)

    # Tampilkan tabel hasil
    st.subheader("Forecast Results")
    st.dataframe(preds_df.assign(pred=lambda d: d['pred'].round(2)))

    # Ambil histori untuk plotting
    hist = processed[
        (processed['Product_Code'] == prod_sel) &
        (processed['Warehouse'] == wh_sel)
    ][['Date', 'sales']].sort_values('Date')

    # Gabungkan histori + forecast untuk grafik
    plot_df = pd.concat([
        hist.rename(columns={'sales': 'value'}),
        preds_df.rename(columns={'pred': 'value'})
    ])

    # Plot line chart
    st.line_chart(plot_df.set_index('Date')['value'])

    st.success("Done.")
