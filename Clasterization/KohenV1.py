import numpy as np
import random
import matplotlib.pyplot as plt


# Функция для инициализации весов SOM
def initialize_weights(input_dim, output_dim):
    weights = np.random.rand(output_dim[0], output_dim[1], input_dim)
    return weights


# Функция для определения победителя (наиболее близкого нейрона) в SOM
def find_best_matching_unit(data_point, weights):
    min_distance = float('inf')
    best_unit = (0, 0)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            distance = np.linalg.norm(data_point - weights[i, j])
            if distance < min_distance:
                min_distance = distance
                best_unit = (i, j)
    return best_unit


# Функция для обновления весов SOM
def update_weights(weights, data_point, learning_rate, best_unit, radius):
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            distance = np.linalg.norm(np.array(best_unit) - np.array([i, j]))
            if distance <= radius:
                weights[i, j] += learning_rate * (data_point - weights[i, j])


# Функция для обучения SOM
def train_som(data, output_dim, num_epochs, initial_learning_rate, initial_radius):
    input_dim = data.shape[1]
    weights = initialize_weights(input_dim, output_dim)

    for epoch in range(num_epochs):
        learning_rate = initial_learning_rate * (1 - epoch / num_epochs)
        radius = initial_radius * (1 - epoch / num_epochs)

        for data_point in data:
            best_unit = find_best_matching_unit(data_point, weights)
            update_weights(weights, data_point, learning_rate, best_unit, radius)

    return weights


# Ваши данные
data = np.array([[2, 2, 0],
                 [2, 3, 2],
                 [1, 3, 0],
                 [3, 1, 2],
                 [1, 1, 0],
                 [1, 2, 0],
                 [2, 2, 0],
                 [3, 2, 2]])

# Параметры SOM
output_dim = (3, 3)  # Размерность выходного слоя SOM
num_epochs = 100  # Количество эпох обучения
initial_learning_rate = 0.1  # Начальная скорость обучения
initial_radius = max(output_dim) / 2  # Начальный радиус

# Обучение SOM
weights = train_som(data, output_dim, num_epochs, initial_learning_rate, initial_radius)

# Поиск победителей (кластеров) для каждой точки данных
clusters = []
for data_point in data:
    best_unit = find_best_matching_unit(data_point, weights)
    clusters.append(best_unit)

# Визуализация результатов
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b']  # Цвета для каждого кластера
for i, cluster in enumerate(clusters):
    ax.scatter(data[i, 0], data[i, 1], data[i, 2], c=colors[cluster[i] * output_dim[1]], marker='o')

ax.set_xlabel('Признак 1')
ax.set_ylabel('Признак 2')
ax.set_zlabel('Признак 3')
plt.show()

