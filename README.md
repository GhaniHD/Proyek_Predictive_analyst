Berikut adalah contoh penulisan README untuk **Laporan Proyek Machine Learning - Ghani Husna Darmawan** berdasarkan informasi yang kamu berikan:

---

# Laporan Proyek Machine Learning

## Prediksi Penjualan Mingguan Walmart

**Oleh:** Ghani Husna Darmawan

---

## Domain Proyek

Industri ritel sangat bergantung pada kemampuan memprediksi penjualan untuk mengoptimalkan inventaris, strategi promosi, dan perencanaan keuangan. Walmart, sebagai salah satu jaringan ritel terbesar di dunia, membutuhkan pemahaman pola penjualan dari waktu ke waktu agar dapat menghindari kekurangan stok, kelebihan stok, atau kerugian finansial.

Dengan menggunakan data penjualan historis, pendekatan Machine Learning, khususnya time series forecasting, dapat digunakan untuk memprediksi penjualan mingguan di masa depan. Proyek ini berfokus pada pembuatan model yang mampu memproyeksikan tren penjualan berdasarkan data historis tersebut.

### Referensi

* Zhang, G., Eddy Patuwo, B., & Hu, M. Y. (1998). *Forecasting with artificial neural networks: The state of the art*. International journal of forecasting, 14(1), 35–62.
* Taylor, S. J., & Letham, B. (2018). *Forecasting at scale*. The American Statistician, 72(1), 37–45.

---

## Business Understanding

### Problem Statements

* Bagaimana memprediksi total penjualan mingguan Walmart di masa depan berdasarkan data historis?
* Model peramalan mana yang terbaik untuk memproyeksikan tren penjualan mingguan?

### Goals

* Menghasilkan model prediktif yang mampu memproyeksikan penjualan mingguan Walmart secara akurat.
* Menentukan model terbaik berdasarkan metrik evaluasi akurasi prediksi deret waktu.

### Solution Statements

* Membangun model time series forecasting menggunakan Prophet dan SARIMA untuk memodelkan tren dan musiman data penjualan.
* Melakukan tuning parameter untuk meningkatkan akurasi model.
* Mengevaluasi model menggunakan metrik MAE, RMSE, dan MAPE.

---

## Data Understanding

Dataset berasal dari Kaggle: *Walmart Weekly Sales Dataset* yang terdiri dari tiga file:

* `stores.csv` : informasi 45 toko Walmart
* `features.csv` : fitur eksternal mingguan seperti suhu, harga bahan bakar, promosi
* `sales.csv` : data penjualan mingguan per toko dan departemen

Fokus utama adalah pada file `sales.csv` dengan kolom:

* Store : ID toko
* Dept : ID departemen
* Date : tanggal
* Weekly\_Sales : total penjualan mingguan
* IsHoliday : indikator minggu libur nasional

**Exploratory Data Analysis (EDA):**

* Jumlah data: ±421.570 baris
* Jumlah toko: 45
* Rentang tanggal: 2010 – 2012
* Fluktuasi penjualan tinggi menjelang libur seperti Thanksgiving dan Natal

---

## Data Preparation

* Menggabungkan ketiga file CSV berdasarkan Store, Date, dan Dept.
* Agregasi data berdasarkan Date untuk prediksi total penjualan mingguan seluruh toko.
* Mengubah tanggal ke format datetime dan menjadikannya indeks.
* Mengurutkan data berdasarkan waktu.
* Menghapus outlier Weekly\_Sales menggunakan metode IQR filtering.
* Memeriksa dan mengisi nilai null jika ada.
* Visualisasi tren dan musiman mingguan.

---

## Modeling

### Model 1: Prophet

* Parameter utama: `yearly_seasonality=True`, `weekly_seasonality=True`, `changepoint_prior_scale=0.05`
* Input data format: ds (datetime), y (target)
* Kelebihan: mudah digunakan, menangani musiman dan libur eksplisit
* Kekurangan: sensitif terhadap data tidak stasioner/sparse

### Model 2: SARIMA

* Parameter: (p,d,q)(P,D,Q,m) dengan periode musiman m=52 (mingguan)
* Pemilihan parameter optimal dengan AIC dan grid search
* Kelebihan: akurat untuk pola musiman kuat, statistik dan interpretatif
* Kekurangan: butuh data stasioner, pemilihan parameter kompleks

---

## Evaluation

### Metrik yang digunakan:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Square Error)
* MAPE (Mean Absolute Percentage Error)

### Hasil Evaluasi:

![image](https://github.com/user-attachments/assets/073c74d3-4230-451c-8ccf-1e7aead753a5)


Model SARIMA menunjukkan performa lebih baik dibanding Prophet berdasarkan seluruh metrik evaluasi, sehingga dipilih sebagai model terbaik untuk prediksi penjualan mingguan Walmart.

---
