# 🚫 Deteksi Komentar Toxic (UAP Machine Learning)

**Nama:** Daffa Nugroho  
**Prodi:** Informatika - Universitas Muhammadiyah Malang  
**Semester:** 7 (Peminatan Data Science)

---

## 📌 Deskripsi Proyek
Proyek ini dibuat untuk memenuhi **Ujian Akhir Praktikum (UAP) Machine Learning**. Tujuan utama sistem ini adalah mengklasifikasikan komentar pada sosial media apakah termasuk kategori **Toxic** (Ujaran Kebencian/Kasar) atau **Non-Toxic** (Aman).

Sistem ini membandingkan tiga arsitektur model Deep Learning:
1.  **LSTM (Long Short-Term Memory):** Sebagai *Base Model*.
2.  **IndoBERT (Indobenchmark):** Model *Pretrained* Transfer Learning (BERT Base).
3.  **DistilBERT Multilingual:** Model *Pretrained* versi ringan (Distilled).

Aplikasi dideploy menggunakan **Streamlit** agar mudah digunakan oleh pengguna awam.

## 📂 Dataset
Dataset yang digunakan adalah **Indonesian Abusive and Hate Speech Twitter Text** yang terdiri dari 13.000+ data tweet.
* **Preprocessing:** Case folding, cleaning simbol/angka, stopword removal.
* **Labeling:**
    * `1` (Toxic): Jika mengandung Hate Speech atau Abusive.
    * `0` (Non-Toxic): Jika kalimat bersih.

## 📊 Hasil Evaluasi & Perbandingan Model
Berdasarkan pengujian pada data test (20%), berikut adalah hasil akurasinya:

| Peringkat | Nama Model | Akurasi | Keterangan |
| :--- | :--- | :--- | :--- |
| 🥇 **1** | **IndoBERT** | **98.00%** | **Model Terbaik.** Sangat cerdas memahami konteks bahasa gaul Indonesia. |
| 🥈 **2** | **DistilBERT** | **98.00%** | Akurasi setara IndoBERT namun proses training lebih cepat. |
| 🥉 **3** | **LSTM** | **84.24%** | Cukup baik untuk model dasar, namun kalah jauh dibanding Pretrained. |

**Kesimpulan:**
Metode **Transfer Learning (IndoBERT/DistilBERT)** terbukti jauh lebih efektif dibandingkan model konvensional (LSTM) untuk kasus klasifikasi teks bahasa Indonesia.

## 🚀 Cara Menjalankan Project (Local)

Karena keterbatasan ukuran file GitHub (Limit 100MB), **File Model (.pth / .bin) TIDAK disertakan** dalam repository ini. Anda perlu menjalankannya secara lokal.

1.  **Clone Repository**
    ```bash
    git clone [https://github.com/daffa110404/UAPDAFFA.git](https://github.com/daffa110404/UAPDAFFA.git)
    cd UAPDAFFA
    ```

2.  **Install Library**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Generate Model (Wajib)**
    Jalankan notebook berikut satu per satu untuk menghasilkan file model di folder `models/`:
    * Buka folder `Code/`
    * Run `3_training_bert.ipynb` (Untuk membuat model IndoBERT)
    * Run `4_training_distilbert.ipynb` (Untuk membuat model DistilBERT)
    * Run `2_training_lstm.ipynb` (Untuk membuat model LSTM)

4.  **Jalankan Aplikasi Web**
    Kembali ke terminal utama, jalankan:
    ```bash
    streamlit run app.py
    ```

## 🖼️ Tampilan Aplikasi
Aplikasi memiliki fitur:
* Input teks komentar bebas.
* Pilihan Model (IndoBERT, DistilBERT, LSTM).
* Visualisasi tingkat keyakinan (Confidence Score).