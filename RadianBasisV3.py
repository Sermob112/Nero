import numpy as np
import matplotlib.pyplot as plt
def radial_basis_function(x, centers, widths):
    return np.exp(-(x - centers)**2 / (2 * widths**2))

def train_rbf_network(x_train, y_train, n_centers, learning_rate, training_epochs):
    # Инициализация центров и ширины радиальных базисных функций
    centers = np.linspace(-5, 5, n_centers)
    widths = np.ones(n_centers) * 0.5

    # Генерация матрицы признаков с радиально базисными функциями
    phi = radial_basis_function(x_train[:, np.newaxis], centers, widths)

    # Инициализация случайных весов
    weights = np.random.randn(n_centers)

    # Обучение модели
    for epoch in range(training_epochs):
        # Вычисление предсказаний модели
        y_pred = np.dot(phi, weights)

        # Вычисление ошибки
        error = y_train - y_pred

        # Обновление весов
        delta_weights = learning_rate * np.dot(phi.T, error)
        weights += delta_weights

    return centers, widths, weights

# Генерация обучающих данных
x_train = np.linspace(-5, 5, 1000)
y_train = np.sqrt(np.abs(x_train**2))

# Задание параметров нейросети
n_centers = 10
learning_rate = 0.01
training_epochs = 100

# Обучение модели
centers, widths, weights = train_rbf_network(x_train, y_train, n_centers, learning_rate, training_epochs)

# Генерация тестовых данных
x_test = np.linspace(-5, 5, 100)
y_test = np.sqrt(np.abs(x_test**2))

# Вычисление предсказаний модели для тестовых данных
phi_test = radial_basis_function(x_test[:, np.newaxis], centers, widths)
y_pred = np.dot(phi_test, weights)

# Вывод результатов
plt.plot(x_test, y_pred, label='Предсказанные значения')
plt.plot(x_test, y_test, label='Истинные значения')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()