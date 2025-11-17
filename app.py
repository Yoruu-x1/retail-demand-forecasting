# app.py
import streamlit as st
import pandas as pd
import json
import joblib
import os
from datetime import datetime

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(ARTIFACTS_DIR, "encoders.json")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "features_order.json")

# Load model + encoders + feature order
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found: artifacts/model.pkl. Jalankan train.py dulu.")
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as fh:
        features = json.load(fh)
    with open(ENCODERS_PATH, "r", encoding="utf-8") as fh:
        encoders = json.load(fh)
    return model, features, encoders

try:
    model, FEATURES_ORDER, ENCODERS = load_artifacts()
except Exception as e:
    st.error("Artifacts load error: " + str(e))
    st.stop()

st.title("Retail Demand Forecasting")
st.markdown(
    """
    **Petunjuk singkat**:
    - Masukkan detail produk dan kondisi toko.
    - Pilih tanggal
    - `StateHoliday` values:
      - **0** = Bukan hari libur (hari kerja biasa)
      - **a** = Hari libur nasional (contoh: Tahun Baru, Hari Kemerdekaan)
      - **b** = Hari besar keagamaan (contoh: Natal, Idul Fitri)
      - **MISSING** = data tidak tersedia
    """
)

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        product_code = st.text_input("Product Code", value="Product_0033")
        warehouse = st.text_input("Warehouse", value="Whse_S")
        product_category = st.text_input("Product Category", value="Category_005")
        stateholiday = st.selectbox("StateHoliday", options=["0", "a", "b", "MISSING"])

    with col2:
        open_flag = st.selectbox("Open (1 buka / 0 tutup)", [1, 0], index=0)
        promo = st.selectbox("Promo (1 = ada)", [0, 1], index=0)
        school_holiday = st.selectbox("SchoolHoliday (1 = libur sekolah)", [0, 1], index=0)
        petrol_price = st.number_input("Petrol_price (contoh: 90)", min_value=0, value=90, step=1)

    date_input = st.date_input("Tanggal (dipakai utk Year/Month/Day/DayOfWeek)", value=datetime(2016,1,5))

    submit = st.form_submit_button("Predict")

if submit:
    # build input dict with exact feature order used in training
    Year = date_input.year
    Month = date_input.month
    Day = date_input.day
    DayOfWeek = date_input.weekday()

    # start with zeros
    input_dict = {f: 0 for f in FEATURES_ORDER}

    # fill numeric/date fields (names must match those in FEATURES_ORDER)
    for k in ["Open", "Promo", "SchoolHoliday", "Petrol_price", "Year", "Month", "Day", "DayOfWeek"]:
        if k in input_dict:
            input_dict[k] = {
                "Open": int(open_flag),
                "Promo": int(promo),
                "SchoolHoliday": int(school_holiday),
                "Petrol_price": float(petrol_price),
                "Year": int(Year),
                "Month": int(Month),
                "Day": int(Day),
                "DayOfWeek": int(DayOfWeek)
            }[k]

    # map categoricals using saved encoders mapping; fallback to mode if unseen
    for cat_col, user_val in [
        ("Product_Code", product_code),
        ("Warehouse", warehouse),
        ("Product_Category", product_category),
        ("StateHoliday", stateholiday)
    ]:
        if cat_col not in input_dict:
            continue
        enc = ENCODERS.get(cat_col)
        if enc is None:
            # no encoder saved (shouldn't happen) -> treat as 0
            input_dict[cat_col] = 0
        else:
            mapping = enc["mapping"]
            mode_string = enc.get("mode")
            # if exact user_val in mapping use it, else fallback to mode_string
            if str(user_val) in mapping:
                input_dict[cat_col] = mapping[str(user_val)]
            else:
                # fallback
                fallback = mode_string if mode_string in mapping else list(mapping.keys())[0]
                input_dict[cat_col] = mapping.get(str(fallback), 0)

    # build dataframe in exact order
    X_input = pd.DataFrame([input_dict], columns=FEATURES_ORDER)

    st.write("### Final input (sent to model)")
    st.dataframe(X_input.T.rename(columns={0: "value"}))

    # predict
    try:
        pred = model.predict(X_input)[0]


        pred_display = abs(pred)

        st.success(f"Predicted Order Demand: **{pred_display:,.2f}** units")

    except Exception as e:
        st.error("Model prediction error:")
        st.code(str(e))
