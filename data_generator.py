import numpy as np
from sklearn.preprocessing import MinMaxScaler


def generate_data(n_samples=1000, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    X = np.sort(15 * np.random.rand(n_samples, 1), axis=0)
    X_flat = X.ravel()

    y = ((np.sin(X_flat) + np.cos(2 * X_flat)) + np.random.normal(0, 0.25, n_samples))

    return X, y


def normalize_data(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler