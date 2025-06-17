# TensorFlow Certificate Training

Repositori ini berisi kumpulan code sebagai persiapan dalam sertifikasi TensorFlow dengan serangkaian model *machine learning*, yang menggunakan TensorFlow dan Keras.
Setiap skrip menggunakan metode yang berbeda, mulai dari regresi linear sederhana hingga topik yang lebih lanjut seperti *convolutional neural networks* (CNN), *natural language processing* (NLP), dan peramalan data deret waktu (*time series forecasting*).

## Ringkasan

Setiap skrip (`Problem_B*.py`) adalah solusi mandiri yang akan mengunduh datasetnya sendiri, membangun model jaringan saraf, melatihnya, dan menyimpan model yang telah dilatih sebagai berkas HDF5 (`.h5`).

---

### B1: Regresi Linear
- **Berkas:** `Problem_B1.py`
- **Tujuan:** Melatih sebuah jaringan saraf sederhana untuk menemukan hubungan linear ($Y = 2X + 3$) antara dua array, `X` dan `Y`.
- **Konsep Utama:** Model `Sequential` dasar, layer `Dense`, *loss function* `mean_squared_error`, layer `Normalization`.
- **Arsitektur Model:** Satu layer `Normalization` diikuti oleh satu layer `Dense` dengan satu unit neuron dan `input_shape` sebesar `[1]`.
- **Dataset:** Dua array `numpy` yang sudah didefinisikan di dalam kode.
- **Target Kinerja:** *Mean Squared Error* (MSE) di bawah `1e-3`.

---

### B2: Klasifikasi Gambar (Fashion MNIST)
- **Berkas:** `Problem_B2.py`
- **Tujuan:** Membangun sebuah *Convolutional Neural Network* (CNN) untuk mengklasifikasikan 10 kelas item pakaian dari dataset Fashion MNIST.
- **Konsep Utama:** CNN, layer `Conv2D` dan `MaxPooling2D`, klasifikasi *multi-class*, *loss function* `sparse_categorical_crossentropy`.
- **Arsitektur Model:**
  - `Conv2D` (32 filter) -> `MaxPooling2D`
  - `Conv2D` (64 filter) -> `MaxPooling2D`
  - `Flatten` -> `Dense` (128 unit) -> `Dense` (10 unit, aktivasi softmax)
- **Dataset:** Dataset Fashion MNIST dari `tf.keras.datasets`.
- **Target Kinerja:** `accuracy` dan `validation_accuracy` di atas 83%.

---

### B3: Klasifikasi Gambar (Rock-Paper-Scissors)
- **Berkas:** `Problem_B3.py`
- **Tujuan:** Membangun sebuah CNN untuk mengklasifikasikan gambar 'Batu', 'Kertas', atau 'Gunting'.
- **Konsep Utama:** CNN, `ImageDataGenerator` untuk augmentasi dan pelabelan data secara otomatis, klasifikasi *multi-class*.
- **Arsitektur Model:**
  - `Conv2D` (64 filter) -> `MaxPooling2D`
  - `Conv2D` (64 filter) -> `MaxPooling2D`
  - `Conv2D` (128 filter) -> `MaxPooling2D`
  - `Conv2D` (128 filter) -> `MaxPooling2D`
  - `Flatten` -> `Dropout` -> `Dense` (512 unit) -> `Dense` (3 unit, aktivasi softmax)
- **Dataset:** Dataset Rock-Paper-Scissors oleh Laurence Moroney.
  ```sh
  [https://github.com/dicodingacademy/assets/releases/download/release-rps/rps.zip](https://github.com/dicodingacademy/assets/releases/download/release-rps/rps.zip)
  ```
- **Target Kinerja:** `accuracy` dan `validation_accuracy` di atas 83%.

---

### B4: Klasifikasi Teks (NLP)
- **Berkas:** `Problem_B4.py`
- **Tujuan:** Membangun sebuah *multi-class classifier* untuk mengkategorikan artikel berita dari dataset BBC ke dalam 5 topik.
- **Konsep Utama:** *Natural Language Processing* (NLP), tokenisasi teks, *padding sequences*, layer `Embedding`, `Conv1D`, klasifikasi teks.
- **Arsitektur Model:** `Embedding` -> `Conv1D` -> `GlobalAveragePooling1D` -> `Flatten` -> `Dense` (24 unit) -> `Dense` (6 unit, aktivasi softmax).
- **Dataset:** Dataset *BBC Text* dari `dicodingacademy/assets`.
- **Target Kinerja:** `accuracy` dan `validation_accuracy` di atas 91%.

---

### B5: Peramalan Data Deret Waktu
- **Berkas:** `Problem_B5.py`
- **Tujuan:** Membangun sebuah model untuk meramalkan suhu maksimum harian berdasarkan data historis.
- **Konsep Utama:** Analisis data deret waktu (*time series*), *windowed datasets*, layer `LSTM`, metrik *Mean Absolute Error* (MAE), *loss function* Huber.
- **Arsitektur Model:**
  - `LSTM` (50 unit, return_sequences=True)
  - `LSTM` (50 unit)
  - `Dense` (10 unit) -> `Dense` (1 unit)
- **Dataset:** `daily-max-temperatures.csv` dari `jbrownlee/Datasets`.
- **Target Kinerja:** *Mean Absolute Error* (MAE) di bawah 0.2 pada dataset yang sudah dinormalisasi.

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
    pip install tensorflow numpy pandas scikit-learn
    ```

## Cara Menjalankan

Setiap skrip dapat dieksekusi langsung dari terminal. Skrip akan menangani pengunduhan dataset yang diperlukan, melatih model, dan menyimpan hasilnya.

Untuk menjalankan solusi dari soal tertentu, gunakan struktur perintah berikut:

```sh
python [nama_berkas]
```
**Contoh:**
```sh
python Problem_B1.py
