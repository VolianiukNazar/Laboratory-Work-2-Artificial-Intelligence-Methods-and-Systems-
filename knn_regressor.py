import numpy as np
from sklearn.metrics import mean_absolute_error


class CustomKNNRegressor:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        predictions = []

        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_values = self.y_train[k_indices]
            predictions.append(np.mean(k_nearest_values))

        return np.array(predictions)


def find_best_k(X_train, y_train, X_test, y_test, max_k=30):
    best_k = 1
    best_mae = float('inf')
    mae_list = []

    for k in range(1, max_k + 1):

        knn = CustomKNNRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mae_list.append(mae)

        if mae < best_mae:
            best_mae = mae
            best_k = k

    return best_k, mae_list


def train_final_model(X_train, y_train, k):
    knn = CustomKNNRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn