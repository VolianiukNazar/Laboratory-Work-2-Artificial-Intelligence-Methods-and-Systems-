import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from data_generator import generate_data, normalize_data
from knn_regressor import find_best_k, train_final_model


def main():
    X, y = generate_data(1000, random_state=5)

    X_scaled, scaler = normalize_data(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=5
    )

    max_k = 30
    best_k, mae_list = find_best_k(X_train, y_train, X_test, y_test, max_k)

    best_model = train_final_model(X_train, y_train, best_k)
    y_pred_test = best_model.predict(X_test)

    accuracy = r2_score(y_test, y_pred_test) * 100

    print(f"Найкраще значення K: {best_k}")
    print(f"Середня абсолютна помилка (MAE): {min(mae_list):.4f}")
    print(f"Точність алгоритму (R^2): {accuracy:.2f}%")


    plt.figure(figsize=(8, 5))
    plt.scatter(X_scaled, y, s=15, color='gray', alpha=0.6)
    plt.title("Крок 1: Вихідні нормалізовані дані")
    plt.xlabel("Нормалізована ознака X")
    plt.ylabel("Цільова змінна y")
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), mae_list, marker='o', color='b', linestyle='dashed')
    plt.title("Крок 2: Залежність помилки від параметра K")
    plt.xlabel("Кількість сусідів (K)")
    plt.ylabel("Помилка (MAE)")
    plt.axvline(best_k, color='r', linestyle='--', label=f'Найкраще K={best_k}')
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    X_plot = np.linspace(X_scaled.min(), X_scaled.max(), 1000)[:, np.newaxis]
    y_plot = best_model.predict(X_plot)
    plt.scatter(X_scaled, y, s=15, color='gray', alpha=0.4, label='Дані')
    plt.plot(X_plot, y_plot, color='red', linewidth=2, label=f'KNN (K={best_k})')
    plt.title("Крок 3: Фінальний результат регресії")
    plt.xlabel("Нормалізована ознака X")
    plt.ylabel("Цільова змінна y")
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
        main()