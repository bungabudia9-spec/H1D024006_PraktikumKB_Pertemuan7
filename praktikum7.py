import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# Muat dataset iris dari file lokal
dataset = pd.read_csv(
    'iris.data',
    header=None
)

# Menyusun data X (fitur) dan y (label)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Mengonversi label dari string menjadi numerik
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Memisahkan dataset menjadi data latih dan data validasi
# rasio 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Membuat model neural network
model = Sequential([
    Input(shape=X_train.shape[1:]),
    Dense(1000, activation='relu'),
    Dense(500, activation='relu'),
    Dense(300, activation='relu'),
    Dense(3, activation='softmax')
])

# Menampilkan summary model
model.summary()

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluasi model
loss, accuracy = model.evaluate(X_test, y_test)

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Visualisasi training
pd.DataFrame(history.history).plot(figsize=(10, 6))
plt.show()

# Prediksi data testing
predictions = model.predict(X_test)

# Mengambil indeks probabilitas tertinggi
predicted_classes = predictions.argmax(axis=1)

print("Prediksi:", predicted_classes)
print("Label Asli:", y_test)

# Membuat confusion matrix
cm = confusion_matrix(y_test, predicted_classes)

# Visualisasi confusion matrix
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

plt.show()

# Fungsi prediksi data baru
def predict_new_data():

    sepal_length = float(input("Masukkan sepal length: "))
    sepal_width = float(input("Masukkan sepal width: "))
    petal_length = float(input("Masukkan petal length: "))
    petal_width = float(input("Masukkan petal width: "))

    # Membuat array data baru
    new_data = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    # Melakukan prediksi
    prediction = model.predict(new_data)

    predicted_class = prediction.argmax(axis=1)

    # Mengubah hasil numerik ke label asli
    predicted_label = label_encoder.inverse_transform(predicted_class)

    print(f"Prediksi kelas: {predicted_label[0]}")

# Menjalankan fungsi prediksi
predict_new_data()