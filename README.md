# Iris Classification with Neural Network

## Deskripsi

Project ini merupakan implementasi **Neural Network menggunakan TensorFlow/Keras** untuk mengklasifikasikan dataset Iris menjadi tiga kelas:

* Iris-setosa
* Iris-versicolor
* Iris-virginica

Model menggunakan 4 fitur utama:

* Sepal length
* Sepal width
* Petal length
* Petal width

---

## Arsitektur Model

Model yang digunakan adalah **Deep Neural Network (DNN)**:

* Input Layer: 4 neuron
* Hidden Layer 1: 1000 neuron (ReLU)
* Hidden Layer 2: 500 neuron (ReLU)
* Hidden Layer 3: 300 neuron (ReLU)
* Output Layer: 3 neuron (Softmax)

**Total parameter: 656,703**

---

## Library

Install dependency berikut:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

---

## Dataset

Dataset Iris disimpan dalam file:

```
iris.data
```

Format data:

```
sepal_length, sepal_width, petal_length, petal_width, class
```

Contoh:

```
5.1,3.5,1.4,0.2,Iris-setosa
6.4,3.2,4.5,1.5,Iris-versicolor
6.3,3.3,6.0,2.5,Iris-virginica
```

---

## Cara Menjalankan

1. Pastikan file berikut ada di folder yang sama:
   * `praktikum7.py`
   * `iris.data`

2. Jalankan dengan perintah:

```bash
py -3.11 praktikum7.py
```

---

## Hasil

* Accuracy: ±96%
* Loss: ±0.07

Model mampu melakukan klasifikasi dengan baik pada data testing.

---

## Output

### Grafik Training

Menampilkan:

* Accuracy
* Validation Accuracy
* Loss
* Validation Loss

### Confusion Matrix

Menunjukkan performa klasifikasi model.

### Prediksi

Contoh output:

```
Prediksi: [1 0 2 ...]
Label Asli: [1 0 2 ...]
```

---

## Prediksi Data Baru

User dapat memasukkan data secara manual:

```
Masukkan sepal length:
Masukkan sepal width:
Masukkan petal length:
Masukkan petal width:
```

Output:

```
Prediksi kelas: Iris-setosa
```

## Identitas
Nama: Bunga Budi Ambarwati
NIM: H1D024006
Praktikum Kecerdasan Buatan  

