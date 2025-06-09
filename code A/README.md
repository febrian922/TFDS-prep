# TensorFlow Certificate Training

Repositori ini berisi kumpulan code sebagai persiapan dalam sertifikasi TensorFlow dengan serangkaian model *machine learning*, yang menggunakan TensorFlow dan Keras.
Setiap skrip menggunakan metode yang berbeda, mulai dari regresi linear sederhana hingga topik yang lebih lanjut seperti *transfer learning*, 
*natural language processing* (NLP), dan peramalan data deret waktu (*time series forecasting*).

## Daftar Isi
* [Ringkasan](#ringkasan)
  * [A1: Regresi Linear](#soal-a1-regresi-linear)
  * [A2: Klasifikasi Gambar (CNN dari Awal)](#soal-a2-klasifikasi-gambar-cnn-dari-awal)
  * [A3: Klasifikasi Gambar (Transfer Learning)](#soal-a3-klasifikasi-gambar-transfer-learning)
  * [A4: Klasifikasi Teks (NLP)](#soal-a4-klasifikasi-teks-nlp)
  * [A5: Peramalan Data Deret Waktu](#soal-a5-peramalan-data-deret-waktu)
* [Prasyarat](#prasyarat)
* [Pengaturan & Instalasi](#pengaturan--instalasi)
* [Cara Menjalankan](#cara-menjalankan)

## Ringkasan

Setiap skrip (`Problem_A*.py`) adalah solusi mandiri yang akan mengunduh datasetnya sendiri, membangun model jaringan saraf, melatihnya, dan menyimpan model yang telah dilatih sebagai berkas HDF5 (`.h5`).

---

### A1: Regresi Linear
- **Berkas:** `Problem_A1.py`
- **Tujuan:** Melatih sebuah jaringan saraf sederhana untuk menemukan hubungan linear ($Y = X + 9$) antara dua array, `x` dan `y`.
- **Konsep Utama:** Model `Sequential` dasar, layer `Dense`, *loss function* `mean_squared_error`.
- **Arsitektur Model:** Satu layer `Dense` dengan satu unit neuron dan `input_shape` sebesar `[1]`.
- **Dataset:** Dua array `numpy` yang sudah didefinisikan di dalam kode.
- **Target Kinerja:** *Mean Squared Error* (MSE) di bawah `1e-4`.

---

### A2: Klasifikasi Gambar (CNN dari Awal)
- **Berkas:** `Problem_A2.py`
- **Tujuan:** Membangun sebuah *Convolutional Neural Network* (CNN) untuk mengklasifikasikan gambar 'kuda' atau 'manusia'.
- **Konsep Utama:** CNN, layer `Conv2D` dan `MaxPooling2D`, `ImageDataGenerator` untuk augmentasi data, klasifikasi biner.
- **Arsitektur Model:**
  - `Conv2D` (16 filter) -> `MaxPooling2D`
  - `Conv2D` (32 filter) -> `MaxPooling2D`
  - `Conv2D` (64 filter) -> `MaxPooling2D`
  - `Flatten` -> `Dropout` -> `Dense` (512 unit) -> `Dense` (1 unit, aktivasi sigmoid)
- **Dataset:** Dataset *Horse or Human* oleh Laurence Moroney.
- **Target Kinerja:** `accuracy` dan `validation_accuracy` di atas 83%.

---

### A3: Klasifikasi Gambar (Transfer Learning)
- **Berkas:** `Problem_A3.py`
- **Tujuan:** Menyelesaikan masalah klasifikasi 'kuda' atau 'manusia' yang sama, namun kali ini menggunakan metode *transfer learning*.
- **Konsep Utama:** *Transfer Learning*, ekstraksi fitur, pembekuan layer (*freezing layers*), model pra-terlatih `InceptionV3`.
- **Arsitektur Model:**
  - **Model Dasar:** Model `InceptionV3` yang telah dilatih sebelumnya tanpa layer klasifikasi di bagian atasnya. Semua layer pada model dasar ini dibekukan (non-trainable).
  - **Kepala Kustom:** Output dari layer `mixed7` pada InceptionV3 dihubungkan ke:
     `Flatten` -> `Dense` (512 unit) -> `Dropout` -> `Dense` (1 unit, aktivasi sigmoid).
- **Dataset:** Dataset *Horse or Human*.
- **Target Kinerja:** `accuracy` dan `validation_accuracy` di atas 97%.

---

### A4: Klasifikasi Teks (NLP)
- **Berkas:** `Problem_A4.py`
- **Tujuan:** Membangun sebuah *binary classifier* untuk menentukan sentimen (positif atau negatif) dari ulasan film pada dataset IMDB.
- **Konsep Utama:** *Natural Language Processing* (NLP), tokenisasi teks, *padding sequences*, layer `Embedding`.
- **Arsitektur Model:** `Embedding` -> `Flatten` -> `Dense` (1 unit, aktivasi sigmoid).
- **Dataset:** Dataset *IMDB Reviews* dari `tensorflow_datasets`.
- **Target Kinerja:** `accuracy` dan `validation_accuracy` di atas 83%.

---

### A5: Peramalan Data Deret Waktu
- **Berkas:** `Problem_A5.py`
- **Tujuan:** Membangun sebuah model untuk meramalkan aktivitas bintik matahari (*sunspots*) berdasarkan data historis.
- **Konsep Utama:** Analisis data deret waktu (*time series*), *windowed datasets*, layer `Conv1D` dan `LSTM`, metrik *Mean Absolute Error* (MAE).
- **Arsitektur Model:**
  - `Conv1D` (64 filter)
  - `LSTM` (64 unit)
  - `LSTM` (64 unit)
  - `Dense` (30 unit) -> `Dense` (1 unit)
- **Dataset:** `sunspots.csv` dari Kaggle.
- **Target Kinerja:** *Mean Absolute Error* (MAE) di bawah 0.15 pada dataset yang sudah dinormalisasi.

## Prasyarat
- Python 3.8+
- Manajer paket `pip`

## Pengaturan & Instalasi

1.  **Kloning repositori ini:**
    ```sh
    git clone [https://github.com/nama-pengguna-anda/nama-repositori-anda.git](https://github.com/nama-pengguna-anda/nama-repositori-anda.git)
    cd nama-repositori-anda
    ```

2.  **Instalasi *library* Python yang dibutuhkan:**
    Cara terbaik adalah menggunakan `requirements.txt`, namun Anda juga bisa menginstalnya secara langsung:
    ```sh
    pip install tensorflow numpy tensorflow-datasets
    ```

## Cara Menjalankan

Setiap skrip dapat dieksekusi langsung dari terminal. Skrip akan menangani pengunduhan dataset yang diperlukan, melatih model, dan menyimpan hasilnya.

Untuk menjalankan solusi dari soal tertentu, gunakan struktur perintah berikut:

```sh
python [nama_berkas]
