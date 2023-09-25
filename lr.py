import numpy as np
import pandas as pd
from math import ceil

# Load Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# Preprocess the dataset
dataset['class'] = pd.Categorical(dataset['class'])
dataset['class'] = dataset['class'].cat.codes

# Definisikan fungsi logistic regression
def logistic_regression(X, y, num_steps, learning_rate):
    """
    Melakukan logistic regression menggunakan gradient descent.

    Args:
        X (numpy.ndarray): Fitur input dengan bentuk (n_samples, n_features).
        y (numpy.ndarray): Nilai target dengan bentuk (n_samples,).
        num_steps (int): Jumlah langkah yang akan dilakukan dalam algoritma gradient descent.
        learning_rate (float): Tingkat pembelajaran untuk algoritma gradient descent.

    Returns:
        numpy.ndarray: Bobot yang dioptimalkan dengan bentuk (n_features,).

    """

    # Tambahkan kolom intercept ke X
    intercept = np.ones((X.shape[0], 1))
    X = np.hstack((intercept, X))

    # Inisialisasi bobot dengan nilai nol
    weights = np.zeros(X.shape[1])

    # Gradient descent
    for step in range(num_steps):
        scores = np.dot(X, weights)
        predictions = 1 / (1 + np.exp(-scores))

        output_error_signal = y - predictions
        gradient = np.dot(X.T, output_error_signal)
        weights += learning_rate * gradient

    return weights

# Set seed untuk generator angka acak
np.random.seed(0)

# Tentukan jumlah lipatan (folds)
k = 5

# Acak dataset
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Bagi dataset menjadi k subset
subset_size = ceil(len(dataset) / k)
subsets = [dataset[i*subset_size:(i+1)*subset_size] for i in range(k)]

# Dapatkan jumlah kelas
num_classes = len(np.unique(dataset['class'].values))

# Terapkan K-Fold Cross Validation
for i in range(k):
    # Bagi subset menjadi set pelatihan dan set pengujian
    train = pd.concat(subsets[:i] + subsets[i+1:])
    test = subsets[i]

    X_train = train.iloc[:, :-1].values
    y_train = train.iloc[:, -1].values
    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    # Terapkan Logistic Regression untuk setiap kelas (One-vs-all)
    weights_all_classes = []
    for c in range(num_classes):
        binary_y_train = np.where(y_train == c, 1, 0)
        weights_c = logistic_regression(X_train, binary_y_train, num_steps=300000, learning_rate=5e-5)
        weights_all_classes.append(weights_c)

    # Cetak koefisien dan intercept untuk setiap lipatan dan setiap kelas
    print('Lipatan ', i+1)
    for c in range(num_classes):
        print('Kelas ', c)
        print('Koefisien: ', weights_all_classes[c][1:])
        print('Intercept: ', weights_all_classes[c][0])

    # Prediksi hasil set pengujian untuk setiap kelas dan pilih kelas dengan skor tertinggi
    final_scores_all_classes = [np.dot(np.hstack((np.ones((X_test.shape[0], 1)), X_test)), weights_c) for weights_c in weights_all_classes]
    preds = np.argmax(final_scores_all_classes, axis=0)

    print('Jumlah sampel yang salah diklasifikasikan: %d' % (y_test != preds).sum())

    # Evaluasi model: hitung akurasi, presisi, recall, dan f1-score untuk setiap lipatan

    TP = np.sum((y_test == 1) & (preds == 1))
    TN = np.sum((y_test == 0) & (preds == 0))
    FP = np.sum((y_test == 0) & (preds == 1))
    FN = np.sum((y_test == 1) & (preds == 0))

    akurasi = (preds == y_test).mean()
    presisi = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * presisi * recall / (presisi + recall)

    print('Akurasi: ', akurasi)
    print('Presisi: ', presisi)
    print('Recall: ', recall)
    print('F1 Score: ', f1_score)
