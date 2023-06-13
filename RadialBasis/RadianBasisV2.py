import numpy as np
import matplotlib.pyplot as plt

# Определение радиально-базисной функции
def radial_basis_function(x, center, width):
    return np.exp(-(x - center)**2 / (2 * width**2))

# Обучающие данные
x_train = np.array([5, 6, 7, 8, 9, 10])
y_train = np.sqrt(np.abs(x_train**2))

# Количество радиально-базисных функций
num_basis_functions =  5

# Инициализация случайных весовых коэффициентов
weights = np.random.randn(num_basis_functions)

# Обучение RBF-сети
learning_rate = 0.01
num_epochs = 1000

for epoch in range(num_epochs):
    # Прямое распространение (forward propagation)
    activations = np.zeros((len(x_train), num_basis_functions))
    for i, x in enumerate(x_train):
        for j in range(num_basis_functions):
            activations[i, j] = radial_basis_function(x, j, 1)

    outputs = np.dot(activations, weights)

    # Вычисление ошибки и градиента
    error = y_train - outputs
    gradient = -2 * np.dot(activations.T, error)

    # Обновление весовых коэффициентов
    weights -= learning_rate * gradient

# Тестирование
x_test = np.linspace(4, 11, 100)  # Равномерно распределенные значения для построения графика
activations_test = np.zeros((len(x_test), num_basis_functions))
for i, x in enumerate(x_test):
    for j in range(num_basis_functions):
        activations_test[i, j] = radial_basis_function(x, j, 1)

predictions = np.dot(activations_test, weights)

# Построение графика
plt.scatter(x_train, y_train, color='blue', label='Обучающие данные')
plt.plot(x_test, predictions, color='red', label='Аппроксимация RBF-сети')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()