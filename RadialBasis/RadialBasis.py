import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt

# Создание данных для обучения и тестирования
x_train = np.arange(0.5, 5, 1)
y_train = np.sqrt(np.abs(x_train**2))

x_test = np.arange(0.1, 5, 0.1)
y_test = np.sqrt(np.abs(x_test**2))

# Определение радиально-базисной функции
def radial_basis_function(x, center, width):
    return np.exp(-(x - center)**2 / (2 * width**2))

# Инициализация параметров радиально-базисной функции
num_rbf_units = 5
centers = np.linspace(0.2, 5, num_rbf_units)
width = 0.45

# Создание матрицы признаков для обучения
rbf_features_train = np.zeros((len(x_train), num_rbf_units))
for i in range(len(x_train)):
    for j in range(num_rbf_units):
        rbf_features_train[i, j] = radial_basis_function(x_train[i], centers[j], width)

# Обучение весов
weights = np.linalg.pinv(rbf_features_train) @ y_train

# Создание матрицы признаков для тестирования
rbf_features_test = np.zeros((len(x_test), num_rbf_units))
for i in range(len(x_test)):
    for j in range(num_rbf_units):
        rbf_features_test[i, j] = radial_basis_function(x_test[i], centers[j], width)

# Предсказание на тестовых данных
y_pred = rbf_features_test @ weights

# Построение графика
plt.plot(x_train, y_train, 'ro', label='Тренеровочные точки')
plt.plot(x_test, y_test, label='Исходная')
plt.plot(x_test, y_pred, label='Полученные')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Аппроксимация функции')
plt.legend()
plt.grid(True)
plt.show()













# x_train = np.arange(0.5, 5, 1)
# y_train = np.sqrt(np.abs(x_train ** 2))
#
# np.random.seed(0)
# weights = np.random.randn(2)
#
#
# # Функция активации ReLU
# def relu(x):
#     return np.maximum(0, x)
#
#
# # Обучение нейронной сети
# learning_rate = 0.01
# num_epochs = 1000
#
# for epoch in range(num_epochs):
#     # Прямой проход (получение предсказаний)
#     outputs = relu(x_train * weights[0])
#
#     # Вычисление функции потерь
#     loss = np.mean((outputs - y_train) ** 2)
#
#     # Обратный проход и оптимизация весов
#     grad_weights = np.array([
#         np.mean(2 * (outputs - y_train) * x_train * (weights[0] * x_train >= 0)),
#         np.mean(2 * (outputs - y_train) * (weights[0] * x_train >= 0))
#     ])
#
#     weights[0] -= learning_rate * grad_weights[0]
#     weights[1] -= learning_rate * grad_weights[1]
#
#     # Вывод информации о процессе обучения
#     if (epoch + 1) % 100 == 0:
#         print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss))
#
# # Тестирование нейронной сети
# x_test = np.arange(0.1, 5, 0.5)
# y_test = relu(x_test * weights[0])
#
# # Построение графика
# plt.plot(x_train, y_train, 'ro', label='Тренеровачный')
# plt.scatter(x_test, y_test, label='Полученный')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('График функции y = sqrt(abs(x^2))')
# plt.legend()
# plt.grid(True)
# plt.show()



# # Создание массива x со значениями от 0 до 10 с шагом 0.1
# x = np.arange(1, 5, 1)
#
# # Вычисление значений функции log(-x)
# y = np.sqrt(np.abs(x**2))
#
#
# print(y)
# print(x)
# # # Создание графика
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('log(-x)')
# plt.title('График функции log(-x) на отрезке [0, 10]')
# plt.grid(True)
# plt.show()