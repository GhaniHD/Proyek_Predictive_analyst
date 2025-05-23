# Laporan Proyek Machine Learning - Ghani Husna Darmawan

## Domain Proyek

Proyek ini berfokus pada permasalahan di sektor retail, khususnya dalam memprediksi penjualan mingguan di Walmart, salah satu jaringan ritel terbesar di Amerika Serikat. Prediksi penjualan sangat penting untuk membantu perusahaan dalam pengelolaan stok, pengadaan barang, dan strategi pemasaran yang lebih efektif. 

Menurut laporan dari McKinsey & Company, perusahaan yang mampu menerapkan prediksi permintaan dengan baik dapat meningkatkan margin keuntungan hingga 2-3% dan mengurangi kelebihan stok hingga 20% \[1]. Oleh karena itu, pengembangan model prediksi penjualan merupakan bagian penting dalam implementasi Machine Learning di dunia bisnis nyata.

---

## Business Understanding

### Problem Statements

- Bagaimana tren penjualan mingguan di Walmart selama periode waktu tertentu?
- Bagaimana korelasi antara fitur numerik seperti suhu, harga bahan bakar, CPI, dan pengangguran terhadap penjualan mingguan?
- Apakah model Machine Learning dapat memprediksi penjualan mingguan secara akurat?

### Goals

- Menganalisis pola tren dan musiman pada data penjualan Walmart.
- Mengidentifikasi faktor-faktor yang memengaruhi volume penjualan mingguan.
- Membangun model time series untuk memprediksi penjualan mingguan di masa depan.

### Solution Statements

- Menggunakan dua algoritma time series forecasting: Prophet dan SARIMAX.
- Memilih model terbaik berdasarkan metrik evaluasi RMSE dan MAE.

---

## Data Understanding

Dataset yang digunakan adalah **Walmart Store Sales Forecasting** yang tersedia di Kaggle:  
[https://www.kaggle.com/datasets/yasserh/walmart-dataset](https://www.kaggle.com/datasets/yasserh/walmart-dataset)

### Variabel yang Digunakan:

| Variabel      | Deskripsi                              |
|---------------|--------------------------------------|
| `Store`       | ID unik masing-masing toko            |
| `Date`        | Tanggal transaksi penjualan           |
| `Weekly_Sales`| Total penjualan mingguan              |
| `Holiday_Flag`| Indikator hari libur                  |
| `Temperature` | Suhu rata-rata mingguan               |
| `Fuel_Price`  | Harga bahan bakar                     |
| `CPI`         | Consumer Price Index                  |
| `Unemployment`| Tingkat pengangguran                  |

### Eksplorasi Awal

- Dataset terdiri dari 6.435 baris dan 8 kolom.
- Tidak terdapat missing value.
  
![Gambar 1: Boxplot untuk mendeteksi outlier pada variabel Weekly_Sales](images/image-1.png)
- Ditemukan beberapa outlier pada kolom `Weekly_Sales` yang dihilangkan menggunakan metode IQR.


### Visualisasi
![Gambar 2: Tren penjualan mingguan menunjukkan pola musiman](images/image-2.png)

Pola Keseluruhan (General Trend):
Secara umum, dari awal 2010 hingga akhir 2012, tidak terlihat tren naik atau turun yang dominan dan konsisten. Garis cenderung berfluktuasi di sekitar nilai rata-rata tertentu, dengan puncak dan lembah yang berulang. Ini menunjukkan bahwa penjualan mingguan relatif stabil dalam jangka panjang selama periode ini, tanpa pertumbuhan atau penurunan yang signifikan secara keseluruhan.

- Pola Musiman (Seasonality):
Ini adalah pola yang paling jelas dan menonjol dalam grafik ini. Kita dapat melihat fluktuasi yang sangat teratur dan berulang setiap tahunnya:

  - Puncak (Peaks):
    Ada puncak penjualan yang signifikan yang terjadi pada akhir tahun (sekitar bulan November-Desember) setiap tahunnya (2010, 2011, 2012). Puncak-puncak ini sangat mencolok, terutama     pada akhir 2010 dan akhir 2011/awal 2012, di mana penjualan mencapai level tertinggi. Ini sangat khas untuk penjualan ritel yang dipengaruhi oleh musim liburan (Natal, Tahun Baru).

  - Lembah (Troughs):
    Sejalan dengan puncak musiman, ada juga lembah penjualan yang terjadi setelah puncak tersebut, biasanya di awal tahun (sekitar Januari-Februari) dan juga di tengah tahun (sekitar       Juli-Agustus). Penjualan cenderung menurun setelah periode liburan dan mungkin mengalami penurunan lagi di musim panas.




![Gambar 3: Heatmap korelasi antara variabel numerik dengan Weekly_Sales](images/image-3.png)

- Weekly_Sales vs. Weekly_Sales: Nilainya 1, seperti yang diharapkan, karena ini adalah korelasi variabel dengan dirinya sendiri.
- Weekly_Sales vs. Temperature: Korelasinya -0.044. Ini adalah korelasi negatif yang sangat lemah, mendekati nol. Artinya, tidak ada hubungan linier yang signifikan antara penjualan mingguan dan suhu.
- Weekly_Sales vs. CPI: Korelasinya -0.07. Juga korelasi negatif yang sangat lemah.
- Temperature vs. CPI: Korelasinya 0.18. Ini menunjukkan korelasi positif yang lemah. Artinya, sedikit peningkatan suhu mungkin sedikit berkorelasi dengan peningkatan CPI, tetapi hubungannya tidak terlalu kuat.
- CPI vs. Unemployment: Korelasinya -0.3. Ini adalah korelasi negatif moderat. Artinya, ketika CPI meningkat, tingkat pengangguran cenderung sedikit menurun, atau sebaliknya. Ini menunjukkan adanya hubungan linier, meskipun tidak sempurna.
- Fuel_Price vs. Unemployment: Korelasinya -0.035. Sangat lemah dan mendekati nol.

- Kesimpulan:
Secara keseluruhan, heatmap ini memungkinkan kita untuk dengan cepat melihat seberapa kuat dan dalam arah apa (positif atau negatif) setiap pasang variabel saling berhubungan secara linier. Ini adalah alat yang sangat berguna dalam analisis data untuk memahami hubungan antar fitur/variabel dalam dataset.


---

## Data Preparation

- Konversi kolom `Date` menjadi format datetime.
- Agregasi total `Weekly_Sales` berdasarkan `Date` untuk membentuk time series tunggal.
- Rename kolom menjadi `ds` dan `y` agar sesuai format input Prophet.
- Urutkan data berdasarkan tanggal.
- Split data untuk evaluasi: 12 minggu terakhir sebagai data test.

![Gambar 4: Data yang sudah disiapkan dan diurutkan berdasarkan tanggal dalam format Prophet](images/image-4.png)

---

## Modeling

### Model 1: Prophet

![Gambar 5: Hasil forecasting model Prophet selama 12 minggu ke depan](images/image-5.png)

Model Prophet telah digunakan untuk memprediksi pola penjualan mingguan di masa depan berdasarkan data historis. Hasilnya dapat dilihat pada grafik "Prophet Forecast", dengan penjelasan sebagai berikut:

- Garis Prediksi (Biru Solid):
Garis biru ini menunjukkan perkiraan penjualan mingguan yang dibuat oleh model.
Model berhasil menangkap pola penjualan yang berulang setiap tahun, seperti puncak penjualan di akhir tahun (musim liburan) dan penurunan di awal atau pertengahan tahun. Ini menunjukkan model sangat baik dalam memahami "musiman" penjualan kita.

- Area Ketidakpastian (Biru Muda):
Area biru muda di sekitar garis prediksi adalah "rentang perkiraan". Ini menunjukkan seberapa yakin model terhadap prediksinya.
Sebagian besar data penjualan aktual (titik hitam) berada dalam rentang ini, yang menandakan bahwa prediksi model cukup akurat.

- Data Penjualan Aktual (Titik Hitam):
Titik-titik hitam adalah data penjualan mingguan yang sebenarnya terjadi di masa lalu.
Meskipun ada beberapa titik yang berada sedikit di luar rentang perkiraan (menunjukkan kejadian tak terduga), secara umum model mampu mengikuti fluktuasi penjualan yang terjadi.

### Model 2: SARIMAX

![Gambar 6: Hasil forecasting model SARIMAX pada periode test](images/image-6.png)

Model SARIMAX telah digunakan untuk memprediksi pola penjualan mingguan. Grafik "SARIMA Forecast" menampilkan data yang digunakan untuk melatih model (Train), data penjualan aktual yang ingin diprediksi (Actual), dan hasil prediksi model (Forecast).

- Data Latih (Biru):
Model dilatih menggunakan data penjualan historis yang menunjukkan pola musiman berulang, seperti peningkatan penjualan di akhir tahun.

- Performa Prediksi (Hijau vs. Oranye):
Pada awal periode prediksi (sekitar Agustus-September 2012), model SARIMAX (garis hijau) cukup baik dalam mengikuti tren penjualan aktual (garis oranye).
Namun, model ini gagal memprediksi puncak penjualan yang signifikan di akhir tahun 2012. Garis prediksi (hijau) menunjukkan nilai yang jauh lebih rendah dibandingkan dengan penjualan aktual (oranye) yang melonjak drastis pada periode liburan.


---

## Evaluation

### Metrik Evaluasi

1. **Mean Absolute Error (MAE)**  
MAE mengukur rata-rata nilai absolut selisih antara prediksi dengan nilai aktual.

  ![alt text](images/image-7.png)

Dimana 𝑦 𝑖 y i ​ adalah nilai aktual dan 𝑦 ^ 𝑖 y ^ ​ i ​ adalah nilai prediksi.

3. **Root Mean Squared Error (RMSE)**
RMSE memberikan penalti lebih besar terhadap error yang besar dengan menghitung akar dari rata-rata kuadrat error.

   ![alt text](images/image-8.png)
   
RMSE sering digunakan dalam forecasting karena lebih sensitif terhadap deviasi besar.

### Hasil Evaluasi
![Gambar 7: Perbandingan metrik evaluasi MAE dan RMSE antara model Prophet dan SARIMAX](images/image.png)

### Kesimpulan

Model **SARIMAX** memberikan performa lebih baik dibandingkan Prophet berdasarkan metrik MAE dan RMSE, sehingga dipilih sebagai model terbaik untuk prediksi penjualan mingguan Walmart.

---

## Referensi

\[1] McKinsey & Company. (2019). *How retailers can drive profitable growth through demand forecasting*.

---

*Catatan: Visualisasi tren, boxplot outlier, dan heatmap korelasi tersedia dalam notebook proyek.*

