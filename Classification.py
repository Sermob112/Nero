import numpy as np
import matplotlib.pyplot as plt

# Генерируем случайные точки для классификации
num_samples = 10
x1 = np.random.normal(2, 1, (num_samples, 2))
x2 = np.random.normal(-2, 1, (num_samples, 2))
inputs = np.vstack((x1, x2))
targets = np.concatenate((np.zeros(num_samples), np.ones(num_samples)))

# Инициализируем веса и смещение случайными значениями
weights = np.random.rand(2)
bias = np.random.rand()


# Определяем функцию активации (сигмоида)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Определяем функцию потерь (бинарная кросс-энтропия)
def binary_cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Обучаем модель
learning_rate = 0.1
num_epochs = 100

for epoch in range(num_epochs):
    # Прямой проход
    outputs = sigmoid(np.dot(inputs, weights) + bias)

    # Вычисляем функцию потерь
    loss = binary_cross_entropy(targets, outputs)

    # Вычисляем градиенты
    dloss_dpred = (outputs - targets) / (outputs * (1 - outputs))
    dpred_dz = outputs * (1 - outputs)
    dz_dw = inputs
    dz_db = 1

    dloss_dw = np.dot(dz_dw.T, dloss_dpred * dpred_dz)
    dloss_db = np.sum(dloss_dpred * dpred_dz)

    # Обновляем веса и смещение
    weights -= learning_rate * dloss_dw
    bias -= learning_rate * dloss_db

# Выводим график точек и прямую классификации
plt.scatter(x1[:, 0], x1[:, 1], c='b', label='Class 0')
plt.scatter(x2[:, 0], x2[:, 1], c='r', label='Class 1')

# Создаем сетку точек для построения прямой
x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Получаем предсказания модели для сетки точек
preds = sigmoid(np.dot(grid_points, weights) + bias)

# Отображаем прямую классификации
plt.contourf(xx, yy, preds.reshape(xx.shape), alpha=0.5, cmap='coolwarm')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()