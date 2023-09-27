# Multiclass Logistic Regression

Implementasi multiclass logistic regression menggunakan Python dengan menggunakan metode gradient descent.

## Deskripsi

Kode ini mengimplementasikan multiclass logistic regression, juga dikenal sebagai multinomial logistic regression atau softmax regression. Multiclass logistic regression digunakan ketika ingin memprediksi lebih dari 2 kelas. Kode ini menjelaskan bagaimana matematika di balik multiclass logistic regression bekerja dan mengimplementasikannya menggunakan metode gradient descent dari awal menggunakan Python.

## Problem Statement

Dalam kasus ini, diasumsikan terdapat N orang/observasi, setiap orang memiliki M fitur, dan mereka termasuk dalam C kelas. Diberikan:

- Matriks X berukuran N x M. Xij mewakili orang ke-i dengan fitur ke-j.
- Vektor Y berukuran N. Yi mewakili orang ke-i yang termasuk dalam kelas ke-k.

Yang tidak diketahui:

- Matriks bobot W berukuran M x C. Wjk mewakili bobot untuk fitur ke-j dan kelas ke-k.

Tujuan kita adalah mencari W dan menggunakan W untuk memprediksi keanggotaan kelas dari observasi X yang diberikan.

## Alur Kerja Multiclass Logistic Regression

Jika kita mengetahui X dan W (misalnya kita memberikan nilai awal W semua 0), berikut adalah alur kerja multiclass logistic regression:

1. Hitung perkalian antara X dan W, kita dapatkan Z = -XW.
2. Hitung softmax untuk setiap baris Zi: Pi = softmax(Zi) = exp(Zi) / ∑Ck=0 exp(Zik). Setiap baris Zi harus merupakan perkalian setiap baris X (yaitu Xi) dan seluruh matriks W. Sekarang setiap baris P harus berjumlah 1.
3. Ambil argmax untuk setiap baris dan temukan kelas dengan probabilitas tertinggi.

## Likelihood

Dalam kasus ini, kita diberikan Y. Jadi untuk observasi yang diberikan, kita tahu kelas dari observasi ini, yaitu Yi. Fungsi likelihood dari Yi diberikan Xi dan W adalah probabilitas observasi i dan kelas k = Yi, yang merupakan softmax dari Zi,k = Yi. Dan fungsi likelihood dari Y diberikan X dan W adalah perkalian semua observasi. 

## Loss Function / Negative Log-Likelihood

Selanjutnya, kita menghitung fungsi loss. Kita menggunakan fungsi negative log-likelihood dan membaginya dengan ukuran sampel. Fungsi loss ini juga ditambahkan dengan regularisasi L2. 

## Gradient

Perhitungan gradien dilakukan dengan menggunakan rumus yang telah diturunkan. Gradien Wk=Yi terhadap Wk adalah matriks identitas I[Yi=k]. 

## Penggunaan

1. Impor library yang diperlukan.
2. Definisikan fungsi one_hot_encode untuk mengubah vektor target menjadi matriks one-hot encoded.
3. Definisikan fungsi softmax untuk menghitung softmax dari matriks Z.
4. Definisikan fungsi loss untuk menghitung fungsi loss untuk logistic regression.
5. Definisikan fungsi gradient untuk menghitung gradien untuk logistic regression.
6. Definisikan fungsi gradient_descent untuk melakukan algoritma gradient descent dengan learning rate dan koefisien regulasi yang tetap.
7. Definisikan kelas Multiclass untuk melakukan training pada model logistic regression.
8. Gunakan dataset iris sebagai contoh.
9. Buat objek model dari kelas Multiclass dan lakukan training dengan menggunakan fit.
10. Lakukan prediksi pada data baru dengan menggunakan predict.
11. Cetak hasil prediksi dan periksa keakuratan prediksi.

## Requirements

- numpy
- pandas
- scikit-learn

## Cara Menjalankan

1. Pastikan semua library yang diperlukan telah diinstal.
2. Jalankan kode di lingkungan Python.
