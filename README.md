Tentu, saya tambahkan keterangan penamaan gambar (caption) di setiap bagian sesuai dengan permintaanmu tanpa menghapus referensi gambar. Berikut versi revisinya dengan penamaan gambar yang jelas:

---

# Laporan Proyek Machine Learning - Ghani Husna Darmawan

## Domain Proyek

Proyek ini berfokus pada permasalahan yang terjadi di sektor retail, khususnya pada prediksi penjualan mingguan di Walmart, yang merupakan salah satu jaringan ritel terbesar dan paling berpengaruh di Amerika Serikat. Prediksi penjualan merupakan aspek yang sangat krusial bagi perusahaan retail, karena dapat memberikan gambaran yang lebih akurat untuk pengelolaan stok barang, pengaturan pengadaan, hingga penyusunan strategi pemasaran yang tepat sasaran. Dengan prediksi yang baik, perusahaan dapat mengantisipasi permintaan konsumen secara lebih efisien dan mengurangi risiko kelebihan atau kekurangan stok.

Menurut laporan dari McKinsey & Company, penerapan metode prediksi permintaan yang akurat dapat membantu perusahaan retail meningkatkan margin keuntungan hingga kisaran 2-3% dan sekaligus mengurangi tingkat overstock hingga 20% \[1]. Hal ini membuktikan bahwa pengembangan model prediksi penjualan bukan hanya sebagai alat analisis data, melainkan juga berkontribusi nyata dalam peningkatan performa bisnis dan daya saing perusahaan di pasar.

## Business Understanding

### Problem Statements

* Bagaimanakah pola dan tren penjualan mingguan yang terjadi di Walmart selama periode waktu yang dianalisis?
* Sejauh mana hubungan atau korelasi antara variabel-variabel numerik seperti suhu rata-rata, harga bahan bakar, indeks harga konsumen (CPI), dan tingkat pengangguran terhadap volume penjualan mingguan?
* Apakah model Machine Learning mampu menghasilkan prediksi penjualan mingguan dengan tingkat akurasi yang dapat diandalkan untuk pengambilan keputusan bisnis?

### Goals

* Melakukan analisis mendalam terhadap pola tren dan faktor musiman yang terdapat dalam data penjualan Walmart.
* Mengidentifikasi dan mengukur pengaruh variabel-variabel eksternal yang dapat mempengaruhi penjualan mingguan.
* Mengembangkan dan menguji model prediksi time series yang mampu memberikan estimasi penjualan mingguan pada periode mendatang dengan tingkat kesalahan seminimal mungkin.

### Solution Statements

* Mengimplementasikan dua pendekatan algoritma time series forecasting, yaitu Prophet dari Facebook dan model statistik SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors).
* Menentukan model terbaik berdasarkan metrik evaluasi utama, yaitu Root Mean Squared Error (RMSE) dan Mean Absolute Error (MAE), yang mengukur seberapa jauh hasil prediksi menyimpang dari nilai aktual.

## Data Understanding

Dataset yang digunakan adalah **Walmart Store Sales Forecasting**, tersedia di platform Kaggle melalui tautan berikut:
[https://www.kaggle.com/datasets/yasserh/walmart-dataset](https://www.kaggle.com/datasets/yasserh/walmart-dataset)

### Variabel yang Digunakan:

| Variabel       | Deskripsi                                    |
| -------------- | -------------------------------------------- |
| `Store`        | ID unik masing-masing toko                   |
| `Date`         | Tanggal transaksi penjualan                  |
| `Weekly_Sales` | Total penjualan dalam satu minggu            |
| `Holiday_Flag` | Indikator apakah minggu tersebut libur       |
| `Temperature`  | Suhu rata-rata pada minggu tersebut          |
| `Fuel_Price`   | Harga bahan bakar rata-rata mingguan         |
| `CPI`          | Consumer Price Index (indeks harga konsumen) |
| `Unemployment` | Tingkat pengangguran                         |

### Eksplorasi Awal:

* Dataset terdiri dari 6435 baris dan 8 kolom.
* Tidak terdapat missing value.
![alt text](images/image-1.png)
* Ditemukan beberapa outlier pada kolom `Weekly_Sales` yang kemudian dihilangkan menggunakan metode IQR.

* Dataset memiliki total 6.435 baris dan 8 kolom variabel.
* Tidak terdapat missing value yang signifikan sehingga proses pembersihan data lebih fokus pada penanganan outlier.
* Dari analisis visual boxplot ditemukan adanya outlier pada variabel `Weekly_Sales`, yang kemudian dilakukan penanganan dengan metode Interquartile Range (IQR) untuk menghilangkan nilai ekstrem agar model tidak bias.

![Gambar 1: Boxplot untuk mendeteksi outlier pada variabel Weekly\_Sales](images/image-1.png)

### Visualisasi:

* Plot garis tren penjualan mingguan memperlihatkan adanya pola musiman dan tren naik turun yang khas.

![Gambar 2: Tren penjualan mingguan menunjukkan pola musiman](images/image-2.png)

* Heatmap korelasi antar variabel numerik mengindikasikan hubungan yang lemah hingga sedang antara fitur-fitur tersebut dengan variabel target penjualan.

![Gambar 3: Heatmap korelasi antara variabel numerik dengan Weekly\_Sales](images/image-3.png)

## Data Preparation

* Kolom `Date` dikonversi ke tipe data datetime agar memudahkan pemrosesan time series.
* Data penjualan mingguan diakumulasikan berdasarkan tanggal untuk membentuk satu seri waktu (time series) yang homogen.
* Kolom hasil agregasi diubah namanya menjadi `ds` (tanggal) dan `y` (nilai target) sesuai format yang diperlukan oleh algoritma Prophet.
* Data diurutkan berdasarkan tanggal secara kronologis.
* Data dibagi menjadi data latih dan data uji, dengan 12 minggu terakhir dijadikan data uji untuk validasi prediksi.

![Gambar 4: Data yang sudah disiapkan dan diurutkan berdasarkan tanggal dalam format Prophet](images/image-4.png)

## Modeling

### Model 1: Prophet

* Prophet adalah model additive forecasting yang dirancang untuk menangani tren non-linear dan pola musiman yang kompleks secara otomatis.
* Model ini diimplementasikan tanpa melakukan hyperparameter tuning secara ekstensif untuk melihat performa dasar dari metode tersebut.
* Forecasting dilakukan untuk periode 12 minggu ke depan berdasarkan data latih yang tersedia.

![Gambar 5: Hasil forecasting model Prophet selama 12 minggu ke depan](images/image-5.png)

### Model 2: SARIMAX

* SARIMAX adalah model statistik yang menggabungkan ARIMA dengan komponen musiman dan variabel eksogen (jika ada).
* Parameter yang dipilih adalah order=(1,1,1) untuk bagian ARIMA, dan seasonal\_order=(1,1,1,52) untuk menangkap pola musiman tahunan (52 minggu).
* Prediksi dilakukan pada data uji dengan periode yang sama yaitu 12 minggu terakhir.

![Gambar 6: Hasil forecasting model SARIMAX pada periode test](images/image-6.png)

## Evaluation

### Metrik Evaluasi

Dua metrik evaluasi utama digunakan untuk menilai performa model prediksi:

1. **Mean Absolute Error (MAE)**
   MAE mengukur rata-rata nilai absolut selisih antara prediksi dengan nilai aktual.

   $$
   MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
   $$

   Dimana $y_i$ adalah nilai aktual dan $\hat{y}_i$ adalah nilai prediksi.

2. **Root Mean Squared Error (RMSE)**
   RMSE memberikan penalti lebih besar terhadap error yang besar dengan menghitung akar dari rata-rata kuadrat error.

   $$
   RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
   $$

   RMSE sering digunakan dalam forecasting karena lebih sensitif terhadap deviasi besar.

### Hasil Evaluasi:

![Gambar 7: Perbandingan metrik evaluasi MAE dan RMSE antara model Prophet dan SARIMAX](images/image.png)

Dari hasil evaluasi, model SARIMAX menunjukkan nilai MAE dan RMSE yang lebih rendah dibandingkan Prophet, yang berarti prediksi SARIMAX lebih dekat ke nilai aktual dan lebih stabil dalam menghadapi fluktuasi musiman.

### Kesimpulan:

Model SARIMAX dengan parameter seasonal\_order yang tepat terbukti memberikan performa terbaik dalam memprediksi penjualan mingguan Walmart dibandingkan dengan Prophet, sehingga model ini direkomendasikan untuk digunakan dalam implementasi prediksi jangka pendek. Pendekatan statistik ini efektif menangkap komponen musiman serta tren data yang ada.

---

**Referensi:**
\[1] McKinsey & Company. (2019). How retailers can drive profitable growth through demand forecasting.

*Catatan: Visualisasi tren, boxplot outlier, dan heatmap korelasi dapat dilihat pada notebook proyek yang menyertai laporan ini.*

---

Kalau kamu mau, saya bisa juga bantu membuat penamaan gambar dengan format berbeda (misalnya nomor halaman atau format APA), tinggal bilang saja!
