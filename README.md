# Kumpulan Solusi Machine Learning dengan TensorFlow

Repositori ini berisi kumpulan solusi untuk berbagai tantangan machine learning yang diimplementasikan menggunakan TensorFlow dan Keras. Problem dibagi menjadi tiga seri (A, B, dan C), masing-masing mencakup tugas-tugas seperti regresi, klasifikasi gambar (termasuk transfer learning), klasifikasi teks, dan peramalan data deret waktu (*time series*).

## Code A

### Problem A1: Regresi Linear Sederhana

* **Tugas**: Regresi.
* **Model**: Jaringan saraf dengan satu layer `Dense`.
* **Dataset**: Array NumPy dengan hubungan linear `y = x + 9`.
* **Target**: MSE < 1e-4.

### Problem A2: Klasifikasi Gambar Kuda atau Manusia

* **Tugas**: Klasifikasi Gambar Biner.
* **Model**: CNN dengan `Conv2D`, `MaxPooling2D`, dan `Dense`.
* **Dataset**: "Horse or Human" oleh Laurence Moroney.
* **Target**: Akurasi dan akurasi validasi > 83%.

### Problem A3: Klasifikasi dengan Transfer Learning

* **Tugas**: Klasifikasi Gambar Biner menggunakan *Transfer Learning*.
* **Model**: InceptionV3 pre-trained dengan *classifier* kustom.
* **Dataset**: "Horse or Human".
* **Target**: Akurasi dan akurasi validasi > 97%.

### Problem A4: Klasifikasi Sentimen Review IMDB

* **Tugas**: Klasifikasi Teks Biner (Analisis Sentimen).
* **Model**: Menggunakan `Embedding`, `Flatten`, dan `Dense`.
* **Dataset**: IMDB reviews dari `tensorflow_datasets`.
* **Target**: Akurasi dan akurasi validasi > 83%.

### Problem A5: Prediksi Time Series Sunspots

* **Tugas**: Prediksi Rangkaian Waktu (*Time Series Forecasting*).
* **Model**: Kombinasi `Conv1D` dan `LSTM`.
* **Dataset**: Sunspots.csv.
* **Target**: MAE < 0.15.

---

## Code B

### Problem B1: Regresi Linear Sederhana

* **Tugas**: Regresi.
* **Model**: Jaringan saraf dengan Normalisasi dan satu layer `Dense`.
* **Dataset**: Array NumPy dengan hubungan linear `Y = 2X + 3`.
* **Target**: MSE < 1e-3.

### Problem B2: Klasifikasi Gambar Fashion MNIST

* **Tugas**: Klasifikasi Gambar Multi-kelas.
* **Model**: CNN dengan `Conv2D`, `MaxPooling2D`, dan `Dense`.
* **Dataset**: Fashion MNIST dari `tf.keras.datasets`.
* **Target**: Akurasi dan akurasi validasi > 83%.

### Problem B3: Klasifikasi Gambar Batu-Gunting-Kertas

* **Tugas**: Klasifikasi Gambar Multi-kelas.
* **Model**: CNN dengan `Conv2D`, `MaxPooling2D`, dan `Dense`.
* **Dataset**: Rock-Paper-Scissors oleh Laurence Moroney.
* **Target**: Akurasi dan akurasi validasi > 83%.

### Problem B4: Klasifikasi Teks Berita BBC

* **Tugas**: Klasifikasi Teks Multi-kelas.
* **Model**: Menggunakan `Embedding`, `Conv1D`, `GlobalAveragePooling1D`, dan `Dense`.
* **Dataset**: BBC-text dataset.
* **Target**: Akurasi dan akurasi validasi > 91%.

### Problem B5: Prediksi Time Series Suhu

* **Tugas**: Prediksi Rangkaian Waktu.
* **Model**: Jaringan saraf dengan layer `LSTM`.
* **Dataset**: Daily Max Temperatures.
* **Target**: MAE < 0.2.

---

## Code C

### Problem C1: Regresi Linear Sederhana

* **Tugas**: Regresi.
* **Model**: Jaringan saraf dengan satu layer `Dense`.
* **Dataset**: Array NumPy dengan hubungan linear `Y = 0.5X + 1`.
* **Target**: MSE < 1e-4.

### Problem C2: Klasifikasi Digit Tulisan Tangan MNIST

* **Tugas**: Klasifikasi Gambar Multi-kelas.
* **Model**: CNN dengan `Conv2D`, `MaxPooling2D`, dan `Dense`.
* **Dataset**: MNIST Handwritten digit dari `tf.keras.datasets`.
* **Target**: Akurasi dan akurasi validasi > 91%.

### Problem C3: Klasifikasi Gambar Kucing vs Anjing

* **Tugas**: Klasifikasi Gambar Biner.
* **Model**: CNN dengan `Conv2D`, `MaxPooling2D`, `Dropout`, dan `Dense`.
* **Dataset**: Cats vs Dogs.
* **Target**: Akurasi dan akurasi validasi > 72%.

### Problem C4: Klasifikasi Teks Sarkasme

* **Tugas**: Klasifikasi Teks Biner.
* **Model**: Menggunakan `Embedding`, `GlobalAveragePooling1D`, dan `Dense`.
* **Dataset**: Sarcasm dataset oleh Rishabh Misra.
* **Target**: Akurasi dan akurasi validasi > 75%.

### Problem C5: Prediksi Time Series Konsumsi Listrik

* **Tugas**: Prediksi Rangkaian Waktu Multivariat.
* **Model**: Jaringan saraf dengan `LSTM`, `Dropout`, dan `Dense`.
* **Dataset**: Individual household electric power consumption.
* **Target**: MAE < 0.1.

---

## Cara Menjalankan

Setiap skrip dapat dieksekusi secara individual. Pastikan Anda telah menginstal semua pustaka yang diperlukan.

```bash
pip install tensorflow numpy pandas scikit-learn tensorflow-datasets keras-preprocessing
```
Untuk menjalankan salah satu skrip dan menyimpan model yang telah dilatih, gunakan perintah berikut di terminal:

```bash

python nama_file_problem.py
```

Model yang berhasil dilatih akan disimpan secara otomatis sebagai file .h5 di direktori yang sama.
