import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

def one_hot_encode(Y):
    """
    Mengubah vektor target menjadi matriks one-hot encoded.
    
    Parameters:
    - Y: Vektor target dengan bentuk (N,), di mana N adalah jumlah sampel.
    
    Returns:
    - Matriks one-hot encoded dengan bentuk (N, C), di mana C adalah jumlah kelas.
    """
    unique_labels = np.unique(Y)
    encoded = np.zeros((Y.shape[0], unique_labels.shape[0]))
    for i, label in enumerate(unique_labels):
        encoded[Y == label, i] = 1
    return encoded

def softmax(Z):
    """
    Menghitung softmax dari matriks Z.
    
    Parameters:
    - Z: Matriks input dengan bentuk (N, C), di mana N adalah jumlah sampel dan C adalah jumlah kelas.
    
    Returns:
    - Matriks softmax dengan bentuk yang sama seperti Z.
    """
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def loss(X, Y, W):
    """
    Menghitung fungsi loss untuk logistic regression.
    
    Parameters:
    - X: Matriks fitur dengan bentuk (N, D), di mana N adalah jumlah sampel dan D adalah jumlah fitur.
    - Y: Matriks one-hot encoded target dengan bentuk (N, C), di mana N adalah jumlah sampel dan C adalah jumlah kelas.
    - W: Matriks bobot dengan bentuk (D, C), di mana D adalah jumlah fitur dan C adalah jumlah kelas.
    
    Returns:
    - Nilai loss.
    """
    Z = - X @ W
    N = X.shape[0]
    return (1 / N * (np.trace(X @ W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1)))))

def gradient(X, Y, W, mu):
    """
    Menghitung gradien untuk logistic regression.
    
    Parameters:
    - X: Matriks fitur dengan bentuk (N, D), di mana N adalah jumlah sampel dan D adalah jumlah fitur.
    - Y: Matriks one-hot encoded target dengan bentuk (N, C), di mana N adalah jumlah sampel dan C adalah jumlah kelas.
    - W: Matriks bobot dengan bentuk (D, C), di mana D adalah jumlah fitur dan C adalah jumlah kelas.
    - mu: Koefisien regulasi.
    
    Returns:
    - Gradien bobot.
    """
    Z = - X @ W
    P = softmax(Z)
    N = X.shape[0]
    return 1 / N * (X.T @ (Y - P)) + 2 * mu * W

def gradient_descent(X, Y, max_iter=1000, eta=0.1, mu=0.01):
    """
    Algoritma gradient descent sederhana dengan learning rate (eta) dan koefisien regulasi (mu) tetap.
    
    Parameters:
    - X: Matriks fitur dengan bentuk (N, D), di mana N adalah jumlah sampel dan D adalah jumlah fitur.
    - Y: Matriks target dengan bentuk (N,), di mana N adalah jumlah sampel.
    - max_iter: Jumlah iterasi maksimum.
    - eta: Learning rate.
    - mu: Koefisien regulasi.
    
    Returns:
    - DataFrame yang berisi langkah-langkah iterasi dan nilai loss pada setiap iterasi.
    - Matriks bobot yang telah diperbarui.
    """
    Y_onehot = one_hot_encode(Y)
    W = np.zeros((X.shape[1], Y_onehot.shape[1]))
    step = 0
    step_lst = [] 
    loss_lst = []
    W_lst = []

    while step < max_iter:
        step += 1
        W -= eta * gradient(X, Y_onehot, W, mu)
        step_lst.append(step)
        W_lst.append(W)
        loss_lst.append(loss(X, Y_onehot, W))

    df = pd.DataFrame({
        'step': step_lst, 
        'loss': loss_lst
    })
    return df, W

class Multiclass:
    def fit(self, X, Y):
        """
        Melakukan training pada model logistic regression.
        
        Parameters:
        - X: Matriks fitur dengan bentuk (N, D), di mana N adalah jumlah sampel dan D adalah jumlah fitur.
        - Y: Matriks target dengan bentuk (N,), di mana N adalah jumlah sampel.
        """
        self.loss_steps, self.W = gradient_descent(X, Y)


    def predict(self, H):
        """
        Melakukan prediksi pada data baru.
        
        Parameters:
        - H: Matriks fitur dengan bentuk (M, D), di mana M adalah jumlah data baru dan D adalah jumlah fitur.
        
        Returns:
        - Array dengan label prediksi.
        """
        Z = - H @ self.W
        P = softmax(Z)
        return np.argmax(P, axis=1)

X = load_iris().data
Y = load_iris().target

model = Multiclass()
model.fit(X, Y)

print(model.predict(X))

print(model.predict(X) == Y)
