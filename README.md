# Retail Demand Forecasting App (MLOps Mini Project)

Project ini membangun sistem machine learning untuk memprediksi *Order_Demand* (jumlah permintaan produk) berdasarkan data historis retail. Sistem terdiri dari:

- **Model machine learning** menggunakan LightGBM (setelah cross-validation).
- **Pipeline preprocessing** yang meng-handle kategorikal & numerikal.
- **Aplikasi Streamlit** untuk prediksi berbasis input user.
- **Automated deployment** melalui Streamlit Cloud.
- **GitHub Version Control** sebagai bagian dari workflow MLOps.

## Fitur Utama

### Model Machine Learning
- Menggunakan **LightGBM Regressor** (model terbaik hasil 5-Fold Cross-Validation).
- Preprocessing mencakup:
  - encoding kategorikal dengan LabelEncoder
  - handling StateHoliday ('0', 'a', 'b')
  - feature engineering sederhana (tanggal → year, month, day, dayofweek)

### Streamlit App
- Aplikasi web interaktif untuk memprediksi Order Demand.
- Input: Product code, Warehouse, Category, Promo, Holiday, Price, dll.
- Output: Prediksi kuantitatif jumlah permintaan.

### MLOps Workflow
- Version control dengan Git.
- Model tersimpan di folder **artifacts/**.
- Deployment otomatis melalui Streamlit Cloud.


